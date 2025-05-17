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
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

# --- Partition helpers ------------------------------------------------------
try:
    # FedLab ships handy partition functions
    from fedlab.utils.dataset.functional import (
        hetero_dir_partition as dirichlet_partition,
        homo_partition as iid_partition,
    )
except ImportError as e:
    raise ImportError(
        "FedLab is required for dataset partitioning. "
        "Install it with `pip install fedlab`."
    ) from e


# --------------------------------------------------------------------------- #
#  PUBLIC API                                                                 #
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Parameters
    ----------
    cfg : DictConfig node loaded from `conf/dataset/*.yaml`
        Required keys (case-insensitive):
            name                e.g. 'cifar10', 'mnist'
            num_clients         int, number of federated participants
            partition.strategy  'iid' | 'dirichlet'
            partition.alpha     float, Dir(α) concentration parameter
        Optional:
            root        dataset cache dir (default /tmp/data)
            batch_size  per-client batch size (default 32)

    Returns
    -------
    trainloaders, valloaders : lists of `torch.utils.data.DataLoader`
        Lists have length = num_clients and keep the same order so
        `trainloaders[cid]` is the peer of `valloaders[cid]`.
    """
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))
    batch_size = getattr(cfg, "batch_size", 32)
    num_clients = cfg.num_clients

    # 1) Load the raw (whole) torchvision dataset ---------------------------
    tfm = transforms.Compose([transforms.ToTensor()])
    name = cfg.name.lower()

    if name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name == "mnist":
        train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
        test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
        tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Fake RGB
        transforms.ToTensor(),])
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    # 2) Build separate label lists for partitioning -------------------------
    train_labels = [lbl for _, lbl in train_ds]
    test_labels = [lbl for _, lbl in test_ds]

    # 3) Slice indices for each client separately ----------------------------
    strat = cfg.partition.strategy.lower()
    alpha = getattr(cfg.partition, "alpha", 0.5)

    if strat == "dirichlet":
        num_classes = len(set(train_labels))
        client_train_idcs = dirichlet_partition(train_labels, num_clients, num_classes, alpha)
        client_test_idcs = dirichlet_partition(test_labels, num_clients, num_classes, alpha)
    elif strat == "iid":
        client_train_idcs = iid_partition(train_labels, num_clients)
        client_test_idcs = iid_partition(test_labels, num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {cfg.partition.strategy}")

    # 4) Wrap in DataLoaders with separate train and validation subsets ------
   # ...existing code...

    trainloaders, valloaders = [], []

    for cid in range(num_clients):
        train_indices = client_train_idcs[cid]
        test_indices = client_test_idcs[cid]
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue  # Skip clients with no data

        tl = DataLoader(
            Subset(train_ds, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        vl = DataLoader(
            Subset(test_ds, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        trainloaders.append(tl)
        valloaders.append(vl)

    return trainloaders, valloaders
