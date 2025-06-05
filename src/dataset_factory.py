"""
dataset_factory.py
~~~~~~~~~~~~~~~~~~
Utilities to load a vision dataset once, partition it into
client-specific subsets (IID or Dirichlet non-IID) and return a list of
DataLoaders so Flower can simulate “virtual clients” on a single machine.
"""

import os, random, threading
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split, ConcatDataset
import torchvision
from torchvision import transforms
from flwr.common.logger import log
from logging import INFO
try:
    from fedlab.utils.dataset.functional import hetero_dir_partition as dirichlet_partition
except ImportError as e:
    raise ImportError("FedLab is required for Dirichlet partitioning. Install it with `pip install fedlab`.") from e

# Single‐slot cache so we only download MNIST once
_raw_dataset_cache = None
_cache_lock = threading.Lock()

def build_shared_dataset(cfg) -> Dataset:
    """
    Carve out a single, class‐balanced G from full MNIST train.
    |G| = share_fraction * |full_train|.
    """
    root = os.path.expanduser(cfg.root)
    tfm = transforms.Compose([transforms.ToTensor()])
    full_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)

    total = len(full_train)                       
    beta = cfg.share_fraction             
    G_size = int(beta * total)
    labels = full_train.targets.tolist()           

    num_classes = len(set(labels))                  
    per_class = G_size // num_classes               
    random.seed(42)
    class_indices = {c: [] for c in range(num_classes)}
    for idx, lbl in enumerate(labels):
        if len(class_indices[lbl]) < per_class:
            class_indices[lbl].append(idx)
        if all(len(v) >= per_class for v in class_indices.values()):
            break

    G_indices = [i for sub in class_indices.values() for i in sub]
    G_dataset = Subset(full_train, G_indices)
    return G_dataset

def build_client_loaders(
    cfg, 
    dataloader_cfg, 
    debug: bool, 
    cid: int, 
    G_dataset: Dataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    For client `cid`:
    1) Reconstruct full_train & full_test.
    2) Exclude G_indices from full_train → D_indices.
    3) Dirichlet‐partition D_indices into K non‐IID subsets.
    4) For this client, take α_dist * |G| random from G_dataset.
    5) Concat(private_ds, G_slice) → train DataLoader.
    6) Build a “private” validation split: 10% of that client’s private slice.
    7) Dirichlet‐partition full_test to get test indices for this client → test loader.
    """
    root = os.path.expanduser(cfg.root)
    tfm = transforms.Compose([transforms.ToTensor()])

    # Full MNIST
    full_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
    full_test  = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)

    # Retrieve G_indices exactly as done in build_shared_dataset
    total = len(full_train)                       
    beta = cfg.share_fraction
    G_size = int(beta * total)
    labels = full_train.targets.tolist()
    num_classes = len(set(labels))
    per_class = G_size // num_classes

    random.seed(42)
    class_indices = {c: [] for c in range(num_classes)}
    for idx, lbl in enumerate(labels):
        if len(class_indices[lbl]) < per_class:
            class_indices[lbl].append(idx)
        if all(len(lst) >= per_class for lst in class_indices.values()):
            break
    G_indices = {i for sub in class_indices.values() for i in sub}

    # 2) Build D_indices = all train indices except G_indices
    D_indices = [i for i in range(total) if i not in G_indices]
    D_labels  = [labels[i] for i in D_indices]

    # 3) Dirichlet‐partition D_indices into num_clients
    client_D_idcs = dirichlet_partition(D_labels, cfg.num_clients, num_classes, cfg.alpha)
    # client_D_idcs is a list of length=K, each element is a list of integer‐positions into D_indices

    client_train_idx = [D_indices[j] for j in client_D_idcs[cid]]

    # 4) Build this client’s private subset
    private_idx = client_train_idx
    if len(private_idx) == 0:
        raise RuntimeError(f"[build_client_loaders] Client {cid} got no private data. Try increasing α or reducing β.")

    private_ds = Subset(full_train, private_idx)

    # 5) Sample α_dist * |G| from G_dataset for **this** client
    alpha_dist = cfg.alpha_dist
    per_client_G = int(alpha_dist * len(G_dataset))
    random.seed(42 + cid)
    client_G_subidxs = random.sample(range(len(G_dataset)), per_client_G)
    G_for_client = Subset(G_dataset, client_G_subidxs)

    # Concat private + shared
    train_combined = ConcatDataset([private_ds, G_for_client])
    if debug:
        # If debug, we only keep up to 200 samples total into train_combined
        train_combined = Subset(train_combined, range(min(len(train_combined), 200)))

    train_loader = DataLoader(
        train_combined,
        batch_size=dataloader_cfg.batch_size,
        shuffle=True,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        drop_last=False,
    )

    # 6) Build a small “validation” split out of **private_ds** (90% train / 10% val)
    val_size = int(0.1 * len(private_ds))
    if val_size > 0:
        train_sub, val_sub = random_split(private_ds, [len(private_ds) - val_size, val_size])
    else:
        train_sub, val_sub = private_ds, private_ds

    val_loader = DataLoader(
        val_sub,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
    )

    # 7) Dirichlet‐partition full_test → client_test_idx
    test_labels = full_test.targets.tolist()
    test_idcs_all = dirichlet_partition(test_labels, cfg.num_clients, num_classes, cfg.alpha)
    client_test_idx = test_idcs_all[cid]
    test_ds = Subset(full_test, client_test_idx)
    if debug:
        test_ds = Subset(test_ds, range(min(len(test_ds), 50)))

    test_loader = DataLoader(
        test_ds,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
    )

    log(INFO, f"[Client {cid}] private={len(private_idx)} + shared={per_client_G} → train={len(train_combined)} | val={len(val_sub)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader