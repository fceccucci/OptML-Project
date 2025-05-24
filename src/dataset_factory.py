"""
dataset_factory.py
~~~~~~~~~~~~~~~~~~
Utilities to load a vision dataset once, partition it into
client-specific subsets (IID or Dirichlet non-IID) and return a list of
DataLoaders so Flower can simulate “virtual clients” on a single machine.
"""

from typing import List, Tuple
import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torchvision
from torchvision import transforms
from logging import INFO, WARN
from flwr.common.logger import log


# --- Partition helpers ------------------------------------------------------
try:
    from fedlab.utils.dataset.functional import (
        hetero_dir_partition as dirichlet_partition,
        homo_partition as iid_partition,
    )
except ImportError as e:
    raise ImportError(
        "FedLab is required for dataset partitioning. "
        "Install it with `pip install fedlab`."
    ) from e


def load_dataset(cfg, debug) -> Tuple[Dataset, Dataset, Dataset]:
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))

    tfm = transforms.Compose([
        transforms.ToTensor()
    ])
    full_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
    val_split = 0.1
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size]
    )
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
    
    if debug:
        train_ds = Subset(train_ds, range(min(len(train_ds), 100)))
        val_ds   = Subset(val_ds,   range(min(len(val_ds),   50)))
        test_ds  = Subset(test_ds,  range(min(len(test_ds),  50)))
    return train_ds, val_ds, test_ds

cache_alpha = 0
cache_train_ds, cache_val_ds, cache_test_ds = None, None, None

def build_dataloaders(cfg, dataloader_cfg, debug) -> Tuple[List[DataLoader], List[DataLoader]]:
    alpha = cfg.alpha

    global cache_train_ds, cache_val_ds, cache_test_ds, cache_alpha
    if cache_train_ds is not None and cache_val_ds is not None and cache_test_ds is not None:
        if cache_alpha == alpha:
            log(INFO, "Used cached dataloaders")
            return cache_train_ds, cache_val_ds, cache_test_ds
        else:
            log(WARN, "Tryed to access wrong cache!")
        
    train_ds, val_ds, test_ds = load_dataset(cfg, debug)
    
    num_clients = cfg.num_clients

    # 2) Extract labels for partitioning ------------------------------------
    train_labels = [lbl for _, lbl in train_ds]
    val_labels = [lbl for _, lbl in val_ds]
    test_labels = [lbl for _, lbl in test_ds]

    # 3) Partition indices by strategy --------------------------------------
    num_classes = len(set(train_labels))
    client_train_idcs = dirichlet_partition(train_labels, num_clients, num_classes, alpha)
    client_val_idcs = dirichlet_partition(val_labels, num_clients, num_classes, alpha)
    client_test_idcs = dirichlet_partition(test_labels, num_clients, num_classes, alpha)
 
    # 4) Wrap in DataLoaders for drichlet ------------------------------------------------
    trainloaders, valloaders, testloaders= [], [], []

    for cid in range(num_clients):
        train_indices = client_train_idcs[cid]
        val_indices = client_val_idcs[cid]
        test_indices = client_test_idcs[cid]

        if len(train_indices) == 0:
            print(f"[WARNING] Skipping client {cid}: no training data")
            continue
        if len(test_indices) == 0:
            print(f"[WARNING] Skipping client {cid}: no test data")
            continue

        train_loader = DataLoader(
            Subset(train_ds, train_indices),
            shuffle=True,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            multiprocessing_context='fork',
            drop_last=False,
        )

        val_loader = DataLoader(
            Subset(val_ds, val_indices),
            shuffle=False,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            multiprocessing_context='fork',
        )

        test_loader = DataLoader(
            Subset(test_ds, test_indices),
            shuffle=False,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            multiprocessing_context='fork',
        )


        print(f"[INFO] Client {cid}: {len(train_indices)} train samples, {len(test_indices)} test samples")
        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        testloaders.append(test_loader)

    print(f"[INFO] Returning {len(trainloaders)} clients with data (out of requested {num_clients})")
    cache_train_ds, cache_val_ds, cache_test_ds = trainloaders, valloaders, testloaders
    cache_alpha = alpha
    return trainloaders, valloaders, testloaders
