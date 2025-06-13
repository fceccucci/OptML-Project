"""pytorchlightning_example: A Flower / PyTorch Lightning app.

This module provides utility functions for federated learning experiments,
including parameter handling, aggregation, reproducibility, data loading,
and device selection.
"""

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
from collections.abc import MutableMapping


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def weighted_avg(results: list[tuple[int, float]]) -> float:
    """
    Aggregate evaluation results obtained from multiple clients using a weighted average.

    Args:
        results (list of (int, float)): Each tuple contains (num_examples, loss).

    Returns:
        float: Weighted average loss.
    """
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def get_parameters(model):
    """
    Extract model parameters as a list of NumPy arrays.

    Args:
        model: PyTorch model.

    Returns:
        List of NumPy arrays representing model parameters.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """
    Set model parameters from a list of NumPy arrays.

    Args:
        model: PyTorch model.
        parameters: List of NumPy arrays.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_transforms(batch):
    """
    Apply transforms to the partition from FederatedDataset.

    Args:
        batch (dict): Batch containing images.

    Returns:
        dict: Batch with images converted to tensors.
    """
    batch["image"] = [transforms.functional.to_tensor(img) for img in batch["image"]]
    return batch

def load_data_test_data_loader(cfg):
    """
    Apply transforms to the partition from FederatedDataset.

    Args:
        batch (dict): Batch containing images.

    Returns:
        dict: Batch with images converted to tensors.
    """
    root = os.path.expanduser(getattr(cfg, "root", cfg.dataset.root))
    tfm = transforms.Compose([transforms.ToTensor()])

    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)

    testloader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        multiprocessing_context="spawn", 
    )
    return testloader


fds = None  # Cache FederatedDataset
def load_data(partition_id, num_partitions, cfg):
    """
    Load federated data partition for a given client.

    Args:
        partition_id (int): Partition/client ID.
        num_partitions (int): Total number of partitions/clients.
        cfg: Hydra config object.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train, val, test loaders)
    """
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
        """Collate function for federated batches (identity)."""

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
    """
    Aggregate evaluation results obtained from multiple clients using a weighted average.

    Args:
        results (list of (int, float)): Each tuple contains (num_examples, loss).

    Returns:
        float: Weighted average loss.
    """
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def standard_aggregate(me: List[Tuple[int, Dict[str, Any]]]):
    """
    Aggregate evaluation results obtained from multiple clients using a weighted average.

    Args:
        results (list of (int, float)): Each tuple contains (num_examples, loss).

    Returns:
        float: Weighted average loss.
    """
    log(INFO, "[AGGREGATE]")
    aggregate_metrics = {}
    for key, _ in me[0][1].items():
        avg = weighted_avg([
            (num_examples, metrics[key])
            for num_examples, metrics in me
        ])
        aggregate_metrics[key] = avg

    for id, (_, metrics) in enumerate(me):
        #TODO this random logs the resulting values
        for key, value in metrics.items():
            aggregate_metrics[f"{key}_{id}"] = value

    return aggregate_metrics


def flatten_dict(dictionary, parent_key='', separator='_'):
    """
    Flatten a nested dictionary.

    Args:
        dictionary (dict): The dictionary to flatten.
        parent_key (str): Prefix for keys.
        separator (str): Separator between keys.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def get_best_device():
    """
    Get the best available device for computation.

    Returns:
        str: "mps", "cuda", or "cpu"
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # return torch.device("mps")
        return "mps"
    elif torch.cuda.is_available():
        # return torch.device("cuda")
        return "cuda"
    else:
        # return torch.device("cpu")
        return "cpu"