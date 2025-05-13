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
        homo_partition       as iid_partition,
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
    root        = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))
    batch_size  = getattr(cfg, "batch_size", 32)
    num_clients = cfg.num_clients

    # 1) load the raw (whole) torch-vision dataset ---------------------------
    tfm = transforms.Compose([transforms.ToTensor()])
    name = cfg.name.lower()
    if name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=tfm)
        test_ds  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name == "mnist":
        train_ds = torchvision.datasets.MNIST(root, train=True,  download=True, transform=tfm)
        test_ds  = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")

    # 2) build a label list for partitioning ---------------------------------
    labels = [lbl for _, lbl in train_ds]

    # 3) slice indices for every client --------------------------------------
    strat  = cfg.partition.strategy.lower()
    alpha  = getattr(cfg.partition, "alpha", 0.5)

    if strat == "dirichlet":
        num_classes = len(set(labels))   
        # FedLab signature: hetero_dir_partition(targets, num_clients, dir_alpha, ...)
        client_idcs = dirichlet_partition(labels, num_clients, num_classes, alpha)
    elif strat == "iid":
        client_idcs = iid_partition(labels, num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {cfg.partition.strategy}")


    # 4) wrap in DataLoaders --------------------------------------------------
    trainloaders, valloaders = [], []
    for cid in range(num_clients):
        tl = DataLoader(
            Subset(train_ds, client_idcs[cid]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        # keep validation distribution ≈ training distribution for this client
        vl = DataLoader(
            Subset(test_ds, client_idcs[cid]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
        trainloaders.append(tl)
        valloaders.append(vl)

    return trainloaders, valloaders
