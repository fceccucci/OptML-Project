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
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
from torchvision import transforms
import random

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

# --------------------------------------------------------------------------- #
#  PUBLIC API                                                                 #
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg) -> Tuple[List[DataLoader], List[DataLoader], torch.utils.data.Dataset]:
    # Load full train + test
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))
    batch_size = getattr(cfg, "batch_size", 32)
    num_clients = cfg.num_clients

    tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)

    # 1) Split into D (clients) and G (shared) by share_fraction β
    total = len(train_ds)
    beta = getattr(cfg.partition, "share_fraction", 0.10)
    G_size = int(beta * total)
    labels = train_ds.targets.tolist()
    num_classes = len(set(labels))
    per_class = G_size // num_classes

    random.seed(getattr(cfg, "seed", 42))
    class_indices = {c: [] for c in range(num_classes)}
    for idx, lbl in enumerate(labels):
        if len(class_indices[lbl]) < per_class:
            class_indices[lbl].append(idx)
        if all(len(v) == per_class for v in class_indices.values()): break

    G_indices = [i for sub in class_indices.values() for i in sub]
    D_indices = [i for i in range(total) if i not in G_indices]
    G_dataset = Subset(train_ds, G_indices)

     # 3) Partition D non-IID via Dirichlet
    D_labels = [labels[i] for i in D_indices]
    partition = dirichlet_partition(D_labels, num_clients, num_classes, getattr(cfg.partition, "alpha", 0.5))
    # FedLab returns a dict mapping client_id to index list
    client_train_indices = []
    for cid in range(num_clients):
        idxs = partition[cid]
        client_train_indices.append([D_indices[j] for j in idxs])

    # 3) Build per-client loaders with α-portion of G
    alpha_dist = getattr(cfg.partition, "alpha_dist", 0.5)
    per_client_G = int(alpha_dist * len(G_dataset))
    trainloaders, valloaders = [], []

    for cid in range(num_clients):
        priv_idxs = client_train_indices[cid]
        if not priv_idxs: continue

        private_ds = Subset(train_ds, priv_idxs)
        random.seed(getattr(cfg, "seed", 42) + cid)
        client_idxs = random.sample(range(len(G_dataset)), per_client_G)
        client_G = Subset(G_dataset, client_idxs)

        combined = ConcatDataset([private_ds, client_G])
        print(f"[INFO] Client {cid}: {len(priv_idxs)} private + {len(client_idxs)} shared = {len(combined)} total train samples")
        trainloaders.append(DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(Subset(test_ds, dirichlet_partition([lbl for _,lbl in test_ds],  # reuse split
                                                          num_clients, num_classes, cfg.partition.alpha)[cid]),
                                    batch_size=batch_size, shuffle=False, num_workers=2))
        

    return trainloaders, valloaders, G_dataset
