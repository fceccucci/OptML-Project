"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from collections import OrderedDict
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from logging import INFO
from flwr.common.logger import log
import torchvision



logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# class LitAutoEncoder(pl.LightningModule):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(),
#             nn.Linear(64, 28 * 28),
#         )

#     def forward(self, x) -> Any:
#         embedding = self.encoder(x)
#         return embedding

#     def configure_optimizers(self) -> Adam:
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx) -> torch.Tensor:
#         x = train_batch["image"]
#         x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         loss = F.mse_loss(x_hat, x)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx) -> None:
#         self._evaluate(batch, "val")

#     def test_step(self, batch, batch_idx) -> None:
#         self._evaluate(batch,# Stolen from strategy implementation!
def weighted_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [transforms.functional.to_tensor(img) for img in batch["image"]]
    return batch

def load_data_test_data_loader(cfg):
    root = os.path.expanduser(getattr(cfg, "root", cfg.dataset.root))
    tfm = transforms.Compose([transforms.ToTensor()])

    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)

    testloader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory
    )
    return testloader


fds = None  # Cache FederatedDataset
def load_data(partition_id, num_partitions, cfg):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, 
                                           partition_by="label", 
                                           alpha=cfg.dataset.alpha)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id, "train")
    partition = partition.with_transform(apply_transforms)
    # 20 % for on federated evaluation
    partition_full = partition.train_test_split(test_size=0.2, seed=42)
    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )

    def federated_collate(batch):
        # batch is a list of length 1 when partition yields whole‐batch dicts,
        # or length N if partition yields single samples; in both cases we
        # just flatten it into one dict:
        # if isinstance(batch, list) and isinstance(batch[0], dict):
        #  batch = batch[0]  # drop the extra list‐of‐dict
        # # now batch["image"] is a Python list of Tensors
        # # images = torch.stack(batch["image"], dim=0)            # (B, C, H, W)
        # # labels = torch.tensor(batch["label"], dtype=torch.long)  # (B,)
        # return batch["image"], batch["label"]
    
        return batch

    trainloader = DataLoader(
        partition_train_valid["train"],
        shuffle=True,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=federated_collate,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        shuffle=False,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=federated_collate,
    )
    testloader = DataLoader(
        partition_full["test"],
        shuffle=False,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=federated_collate,
    )
    return trainloader, valloader, testloader

# Stolen from strategy implementation!
def weighted_loss_avg(results: list[tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def standard_aggregate(me: List[Tuple[int, Dict[str, Any]]]):
    log(INFO, "[AGGREGATE]")
    aggregate_metrics = {}
    for key, _ in me[0][1].items():
        # if "loss" in key:
        #     avg = weighted_loss_avg([
        #         (num_examples, metrics[key])
        #         for num_examples, metrics in me
        #     ])
        # else:
        # TODO we calculate loss and avg with the same function
        avg = weighted_avg([
            (num_examples, metrics[key])
            for num_examples, metrics in me
        ])
        aggregate_metrics[key] = avg

        # aggregate_metrics[f"{key}_dist"] = [metrics[key] for _, metrics in me]

    for id, (_, metrics) in enumerate(me):
        #TODO this random logs the resulting values
        for key, value in metrics.items():
            aggregate_metrics[f"{key}_{id}"] = value

    return aggregate_metrics

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items