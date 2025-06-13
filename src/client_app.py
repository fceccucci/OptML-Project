"""pytorchlightning_example: A Flower / PyTorch Lightning app.

This module defines the client-side logic for federated learning experiments
using PyTorch Lightning and Flower. It includes the FlowerClient class, which
handles local training and evaluation, and the client_fn used to instantiate
clients for each federated participant.
"""
from omegaconf import OmegaConf
import pytorch_lightning as pl
from flwr.client import NumPyClient, ClientApp, Client
from flwr.common import Context
from torch.utils.data import DataLoader
from src.model import SmallCNN
from src.dataset_factory import build_shared_dataset, build_client_loaders
from src.utils import set_parameters, get_parameters, set_seed, get_best_device

class FlowerClient(NumPyClient):
    """A Flower NumPyClient for federated learning with PyTorch Lightning.

    Handles local training and evaluation for a single federated client.
    """
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, cfg):
        """
        Initialize the federated client.

        Args:
            train_loader (DataLoader): DataLoader for the client's training data.
            test_loader (DataLoader): DataLoader for the client's test data.
            cfg: Hydra configuration object.
        """
        self.model = SmallCNN(
            in_channels=1,
            lr=cfg.algorithm.lr
        )
        self.train_loader = train_loader
        # self.val_loader = val_loader # We removed the val loader because we only do like one to five epochs of training here
        self.test_loader = test_loader
        self.cfg = cfg

    def get_parameters(self, config):
        """Get the current model parameters as NumPy ndarrays."""
        return get_parameters(self.model)

    def set_parameters(self, parameters, config=None):
        """Set the model parameters from NumPy ndarrays."""
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        """Train locally (Lightning) on this client’s combined dataset.

        Args:
            parameters: Model parameters to start training from.
            config: Additional configuration.

        Returns:
            Tuple: (updated parameters, number of training examples, training metrics)
        """
        set_parameters(self.model, parameters)
        device = get_best_device()
        self.model.to(device)

        trainer = pl.Trainer(
            max_epochs=self.cfg.algorithm.local_epochs,
            accelerator=device,
            precision=self.cfg.trainer.precision,
            enable_progress_bar=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
        )
        trainer.fit(self.model, train_dataloaders=self.train_loader)
        metrics = {k: v.detach().cpu().item() for k, v in trainer.callback_metrics.items()}
        return get_parameters(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model locally on the test set.

        Args:
            parameters: Model parameters to evaluate.
            config: Additional configuration.

        Returns:
            Tuple: (test loss, number of test examples, test metrics)
        """
        set_parameters(self.model, parameters)
        device = get_best_device()
        self.model.to(device)
        trainer = pl.Trainer(
            enable_progress_bar=False,
            accelerator=device,
            enable_checkpointing=False
        )
        results = trainer.test(self.model, self.test_loader, verbose=False)
        loss = results[0]["test_loss"]
        return loss, len(self.test_loader.dataset), results[0]

def client_fn(context: Context) -> NumPyClient:
    """
    Build and return a federated client for Flower.

    Args:
        context (Context): Flower context object containing the Hydra config.

    Returns:
        NumPyClient: The federated client instance.
    """
    cfg = context.cfg
    cid = int(context.node_config["partition-id"])
    set_seed(42)

    # 1) Rebuild G
    G_dataset = build_shared_dataset(cfg.dataset, cfg.debug)

    # 2) Build this client’s loaders
    train_loader, test_loader = build_client_loaders(
        cfg.dataset, cfg.dataloader, cfg.debug, cid, G_dataset
    )
    return FlowerClient(train_loader, test_loader, cfg).to_client()

app = ClientApp(client_fn=client_fn)
