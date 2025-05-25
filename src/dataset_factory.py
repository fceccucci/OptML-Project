"""
dataset_factory.py
~~~~~~~~~~~~~~~~~~
Utilities to load a vision dataset once, partition it into
client-specific subsets (IID or Dirichlet non-IID) and return a list of
DataLoaders so Flower can simulate “virtual clients” on a single machine.
"""

import threading
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

# single‐slot cache for the raw datasets
_raw_dataset_cache: Tuple = None
_cache_lock = threading.Lock()

def build_dataloaders(
    cfg,
    dataloader_cfg,
    debug: bool,
    cid: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns the (train, val, test) DataLoaders for client `cid`.
    Raw datasets are loaded once and then cached forever.
    Partitioning (dirichlet) still runs each call to reflect cfg.alpha.
    """
    global _raw_dataset_cache

    # 1) Load (or reuse) the raw datasets
    with _cache_lock:
        if _raw_dataset_cache is None:
            train_ds, val_ds, test_ds = load_dataset(cfg, debug)
            _raw_dataset_cache = (train_ds, val_ds, test_ds)
            log(INFO, "Loaded & cached raw datasets")
        else:
            train_ds, val_ds, test_ds = _raw_dataset_cache
            log(INFO, "Reusing cached raw datasets")

    alpha = cfg.alpha
    train_labels = [lbl for _, lbl in train_ds]
    val_labels   = [lbl for _, lbl in val_ds]
    test_labels  = [lbl for _, lbl in test_ds]

    num_clients = cfg.num_clients
    num_classes = len(set(train_labels))
    train_idcs = dirichlet_partition(train_labels, num_clients, num_classes, alpha)
    val_idcs   = dirichlet_partition(val_labels,   num_clients, num_classes, alpha)
    test_idcs  = dirichlet_partition(test_labels,  num_clients, num_classes, alpha)

    train_idx = train_idcs[cid]
    val_idx   = val_idcs[cid]
    test_idx  = test_idcs[cid]

    def make_loader(ds, indices, shuffle):
        return DataLoader(
            Subset(ds, indices),
            shuffle=shuffle,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            drop_last=False if shuffle else False,
        )

    train_loader = make_loader(train_ds, train_idx, shuffle=True)
    val_loader   = make_loader(val_ds,   val_idx,   shuffle=False)
    test_loader  = make_loader(test_ds,  test_idx,  shuffle=False)

    log(INFO, f"[Client {cid}] {len(train_idx)} train / {len(test_idx)} test samples")

    return train_loader, val_loader, test_loader