"""
pytorchlightning_example: A Flower / PyTorch Lightning app.
"""

import pytorch_lightning as pl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
import torch

from src.utils import (
    get_parameters,
    set_parameters,
    standard_aggregate,
    load_data_test_data_loader,
    set_seed,
    get_best_device,
)
from src.model import SmallCNN
from src.straregy_factory import get_fl_algo
from src.server import CustomServer
from src.dataset_factory import build_dataloaders  # <-- IMPORT ADDED
from src.globals import CONFIG_FILE


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp, including one‐time warm-up on G_dataset."""
    cfg = context.cfg
    set_seed(cfg.dataset.seed)

    # -------------------------------------------------------
    # 1) Build G_dataset by asking for client‐0’s slice
    #    (we only care about G_dataset for warm-up)
    # -------------------------------------------------------
    # We pass `cid=0` to get back (train_loader_0, val_loader_0, test_loader_0, G_dataset)
    _, _, _, G_dataset = build_dataloaders(
        cfg.dataset,
        cfg.dataloader,
        cfg.debug,
        cid=0
    )

    # -------------------------------------------------------
    # 2) Warm-up a Lightning SmallCNN on G_dataset
    # -------------------------------------------------------
    device = get_best_device()
    warmup_model = SmallCNN(lr=cfg.algorithm.lr).to(device)
    optimizer = torch.optim.SGD(
        warmup_model.parameters(),
        lr=cfg.algorithm.lr,
        momentum=0.9
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    loaderG = torch.utils.data.DataLoader(
        G_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    warmup_model.train()
    for _ in range(cfg.algorithm.warmup_epochs):
        for x, y in loaderG:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(warmup_model(x), y).backward()
            optimizer.step()

    # -------------------------------------------------------
    # 3) Extract warm-up weights → initial_parameters
    # -------------------------------------------------------
    ndarrays = get_parameters(warmup_model)  # List[np.ndarray]
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # -------------------------------------------------------
    # 4) Prepare a “global_model” instance for evaluate_global()
    # -------------------------------------------------------
    global_model = SmallCNN(lr=cfg.algorithm.lr)

    # Build a test_loader on the entire MNIST test set
    test_loader = load_data_test_data_loader(cfg)

    def evaluate_global(server_rounds, parameters, config):
        """
        Callback to evaluate the global model on the held-out test set
        after each round. Saves the last round’s weights if desired.
        """
        set_parameters(global_model, parameters)
        trainer = pl.Trainer(
            enable_progress_bar=False,
            accelerator=get_best_device(),
            enable_checkpointing=False,
        )
        results = trainer.test(global_model, test_loader, verbose=False)
        loss = results[0]["test_loss"]

        # Optionally save final weights:
        if server_rounds >= (cfg.task.num_of_rounds - 1):
            # Here we save the raw Parameter object; adapt as needed
            torch.save(parameters, "best_model.pt")
        return loss, results[0]

    # -------------------------------------------------------
    # 5) Build the Flower FedAvg strategy using warm-up weights
    # -------------------------------------------------------
    strategy = get_fl_algo(
        cfg,
        initial_parameters=initial_parameters,
        evaluate_global=evaluate_global,
        standard_aggregate=standard_aggregate,
    )

    # -------------------------------------------------------
    # 6) Construct and return ServerAppComponents
    # -------------------------------------------------------
    server = CustomServer(
        strategy=strategy,
        client_manager=SimpleClientManager()
    )
    server_config = ServerConfig(num_rounds=cfg.task.num_of_rounds)
    return ServerAppComponents(server=server, config=server_config)


app = ServerApp(server_fn=server_fn)