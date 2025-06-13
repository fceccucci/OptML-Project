"""pytorchlightning_example: A Flower / PyTorch Lightning app.

This module defines the server-side logic for federated learning experiments
using PyTorch Lightning and Flower. It includes the server function that
initializes the global model, handles optional warmup on shared data, 
defines the global evaluation function, and constructs the Flower server app.
"""
import os, sys
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from omegaconf import OmegaConf
from src.utils import set_seed, get_parameters, set_parameters, standard_aggregate, get_best_device, load_data_test_data_loader
from src.model import SmallCNN
from src.straregy_factory import get_fl_algo
from src.server import CustomServer
from src.dataset_factory import build_client_loaders, build_shared_dataset
import wandb

def server_fn(context: Context) -> ServerAppComponents:
    """
    Build and configure the federated learning server components.

    This function initializes the global model, optionally performs warmup
    training on shared data, defines the global evaluation function, 
    instantiates the federated learning strategy, and returns the server 
    components for Flower.

    Args:
        context (Context): Flower context object containing the Hydra config.

    Returns:
        ServerAppComponents: The server, configuration, and any additional components.
    """
    cfg = context.cfg
    set_seed(42)
    device = get_best_device()

    # 1) Build the shared G
    
    global_model = SmallCNN(lr=cfg.algorithm.lr).to(device)
    test_loader = load_data_test_data_loader(cfg)

    if cfg.dataset.share_fraction > 0 and cfg.algorithm.warmup_epochs > 0:
        G_dataset = build_shared_dataset(cfg.dataset, cfg.debug)
      
        # 2) Warm‐up LightningModel on G for warmup_epochs
        loaderG = torch.utils.data.DataLoader(
            G_dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory
        )
        
        trainer = pl.Trainer(
            max_epochs=cfg.algorithm.warmup_epochs,
            accelerator=device,
            precision=cfg.trainer.precision,
            enable_progress_bar=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
        )

        trainer.fit(global_model, train_dataloaders=loaderG)
        results = trainer.test(global_model, test_loader, verbose=False)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]
        wandb.log({"warmup/loss": loss, "warmup/acc": acc})
    
    # 3) Extract initial parameters (NumPy nd-arrays)
    ndarrays = get_parameters(global_model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # 4) Define evaluate_global
    def evaluate_global(server_rounds, parameters, config):
        """
        Evaluate the global model on the test set.

        Args:
            server_rounds (int): The current federated round.
            parameters: Model parameters to evaluate.
            config: Additional configuration.

        Returns:
            Tuple[float, dict]: The test loss and a dictionary of metrics.
        """
        set_parameters(global_model, parameters)
        trainer = pl.Trainer(enable_progress_bar=False, accelerator=get_best_device(), enable_checkpointing=False,)
        results = trainer.test(global_model, test_loader, verbose=False)
        loss = results[0]["test_loss"]
        if server_rounds >= (cfg.task.num_of_rounds - 1):
            torch.save(parameters, "best_model.pt")
        return loss, results[0]

    # 5) Build Flower strategy
    strategy = get_fl_algo(cfg, initial_parameters, evaluate_global, standard_aggregate)

    # 6) Return components
    server_config = ServerConfig(num_rounds=cfg.task.num_of_rounds)
    custom_server = CustomServer(strategy=strategy, client_manager=SimpleClientManager())
    return ServerAppComponents(server=custom_server, config=server_config)

app = ServerApp(server_fn=server_fn)
