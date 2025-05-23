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

def load_dataset(cfg) -> Tuple[Dataset, Dataset, Dataset]:
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))
    name = cfg.name.lower()
    if name == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        # TODO add val_ds
        test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name == "mnist":
        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
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
    
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}")
    
    return train_ds, val_ds, test_ds


def build_dataloaders(cfg, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> Tuple[List[DataLoader], List[DataLoader]]:
    batch_size = getattr(cfg, "batch_size", 32)
    num_clients = cfg.num_clients

    # 2) Extract labels for partitioning ------------------------------------
    train_labels = [lbl for _, lbl in train_ds]
    val_labels = [lbl for _, lbl in val_ds]
    test_labels = [lbl for _, lbl in test_ds]

    # 3) Partition indices by strategy --------------------------------------
    strat = cfg.partition.strategy.lower()
    alpha = getattr(cfg.partition, "alpha", 0.5)

    if strat == "dirichlet":
        num_classes = len(set(train_labels))
        client_train_idcs = dirichlet_partition(train_labels, num_clients, num_classes, alpha)
        client_val_idcs = dirichlet_partition(val_labels, num_clients, num_classes, alpha)
        client_test_idcs = dirichlet_partition(test_labels, num_clients, num_classes, alpha)
    elif strat == "iid":
        def get_split_sizes(total_len, n_clients):
            base = total_len // n_clients
            leftover = total_len % n_clients
            return [base + 1 if i < leftover else base for i in range(n_clients)]

        train_split_sizes = get_split_sizes(len(train_ds), num_clients)
        val_split_sizes = get_split_sizes(len(val_ds), num_clients)
        test_split_sizes = get_split_sizes(len(test_ds), num_clients)

        train_subsets = torch.utils.data.random_split(train_ds, train_split_sizes)
        val_subsets = torch.utils.data.random_split(val_ds, val_split_sizes)
        test_subsets = torch.utils.data.random_split(test_ds, test_split_sizes)

        trainloaders, valloaders, testloaders = [], [], []
        for cid in range(num_clients):
            if len(train_subsets[cid]) == 0 or len(test_subsets[cid]) == 0:
                print(f"[WARNING] Skipping client {cid}: no data")
                continue

            train_loader = DataLoader(train_subsets[cid], batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_subsets[cid], batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_subsets[cid], batch_size=batch_size, shuffle=False)


            print(f"[INFO] Client {cid}: {len(train_subsets[cid])} train samples, {len(test_subsets[cid])} test samples")
            trainloaders.append(train_loader)
            valloaders.append(val_loader)
            testloaders.append(test_loader)


        print(f"[INFO] Returning {len(trainloaders)} clients with data (out of requested {num_clients})")
        return trainloaders, valloaders, testloaders
    else:
        raise ValueError(f"Unknown partition strategy: {cfg.partition.strategy}")

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
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=False,
        )

        val_loader = DataLoader(
            Subset(test_ds, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        test_loader = DataLoader(
            Subset(test_ds, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        print(f"[INFO] Client {cid}: {len(train_indices)} train samples, {len(test_indices)} test samples")
        trainloaders.append(train_loader)
        valloaders.append(val_loader)
        testloaders.append(test_loader)

    print(f"[INFO] Returning {len(trainloaders)} clients with data (out of requested {num_clients})")
    return trainloaders, valloaders, testloaders
