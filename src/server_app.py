"""pytorchlightning_example: A Flower / PyTorch Lightning app."""
import os, sys
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from omegaconf import OmegaConf
from src.utils import set_seed, get_parameters, set_parameters, standard_aggregate, get_best_device
from src.model import SmallCNN
from src.straregy_factory import get_fl_algo
from src.server import CustomServer
from src.dataset_factory import build_shared_dataset

def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.cfg
    set_seed(42)
    device = get_best_device()

    # 1) Build the shared G
    G_dataset = build_shared_dataset(cfg.dataset)
    
    global_model = SmallCNN(lr=cfg.algorithm.lr).to(device)

    if len(G_dataset) > 0:
        # 2) Warm‐up LightningModel on G for warmup_epochs
        warmup_model = SmallCNN(
            num_classes=cfg.model.num_classes,
            in_channels=1,
            lr=cfg.model.lr
        ).to(device)
    
        loaderG = torch.utils.data.DataLoader(
            G_dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory
        )
        optimizer = torch.optim.SGD(warmup_model.parameters(), lr=cfg.model.lr, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        warmup_model.train()
        for _ in range(cfg.algorithm.warmup_epochs):
            for x, y in loaderG:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss_fn(warmup_model(x), y).backward()
                optimizer.step()
    
        # 3) Extract initial parameters (NumPy nd-arrays)
        initial_nd = get_parameters(warmup_model)
        initial_parameters = ndarrays_to_parameters(initial_nd)
        
    else:
        # No shared data → skip warmup, use cold start
        ndarrays = get_parameters(global_model)
        initial_parameters = ndarrays_to_parameters(ndarrays)

    # 4) Define evaluate_global
    def evaluate_global(server_round: int, parameters, config):
        # load parameters into fresh LightningModel & test on hold‐out MNIST test‐set
        global_model = SmallCNN(
            num_classes=cfg.model.num_classes,
            in_channels=1,
            lr=cfg.model.lr
        ).to(device)
        set_parameters(global_model, parameters)
        trainer = pl.Trainer(
            enable_progress_bar=False,
            accelerator=device,
            enable_checkpointing=False
        )
        test_ds = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                os.path.expanduser(cfg.dataset.root),
                train=False, download=True, transform=transforms.ToTensor()
            ),
            batch_size=cfg.dataloader.batch_size,
            shuffle=False
        )
        results = trainer.test(global_model, test_ds, verbose=False)
        loss = results[0]["test_loss"]
        metrics = {"test_acc": results[0]["test_acc"]}
        # Optionally save best at final round
        if server_round == cfg.task.num_of_rounds:
            torch.save(parameters, "best_model.pt")
        return loss, metrics

    # 5) Build Flower strategy
    strategy = get_fl_algo(cfg, initial_parameters, evaluate_global, standard_aggregate)

    # 6) Return components
    server_config = ServerConfig(num_rounds=cfg.task.num_of_rounds)
    custom_server = CustomServer(strategy=strategy, client_manager=SimpleClientManager())
    return ServerAppComponents(server=custom_server, config=server_config)

app = ServerApp(server_fn=server_fn)
