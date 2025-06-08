"""
dataset_factory.py
~~~~~~~~~~~~~~~~~~
Utilities to load a vision dataset once, partition it into
client-specific subsets (IID or Dirichlet non-IID) and return a list of
DataLoaders so Flower can simulate “virtual clients” on a single machine.
"""

import os, random, threading
from typing import Tuple
from torch.utils.data import DataLoader, Subset, Dataset, random_split, ConcatDataset
import torchvision
from torchvision import transforms
from flwr.common.logger import log
from logging import INFO
try:
    from fedlab.utils.dataset.functional import hetero_dir_partition as dirichlet_partition
except ImportError as e:
    raise ImportError("FedLab is required for Dirichlet partitioning. Install it with `pip install fedlab`.") from e

# single‐slot cache for the raw datasets
_raw_dataset_cache: Tuple = None
_cache_lock = threading.Lock()

def build_shared_dataset(cfg, debug) -> Dataset:
    """
    Carve out a single, class‐balanced G from full MNIST train.
    |G| = share_fraction * |full_train|.
    """
    global _raw_dataset_cache

    # 1) Load (or reuse) the raw datasets
    with _cache_lock:
        if _raw_dataset_cache is None:
            full_train, full_test = load_dataset(cfg, debug)
            _raw_dataset_cache = (full_train, full_test)
            log(INFO, "Loaded & cached raw datasets")
        else:
            full_train, _ = _raw_dataset_cache
            log(INFO, "Reusing cached raw datasets")

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

def load_dataset(cfg, debug) -> Tuple[Dataset, Dataset, Dataset]:
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))

    tfm = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
    
    return train_ds, test_ds

def build_client_loaders(
    cfg, 
    dataloader_cfg, 
    debug: bool, 
    cid: int, 
    G_dataset: Dataset
) -> Tuple[DataLoader, DataLoader]:
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
    global _raw_dataset_cache

    # 1) Load (or reuse) the raw datasets
    with _cache_lock:
        if _raw_dataset_cache is None:
            full_train, full_test = load_dataset(cfg, debug)
            _raw_dataset_cache = (full_train, full_test)
            log(INFO, "Loaded & cached raw datasets")
        else:
            full_train, full_test = _raw_dataset_cache
            log(INFO, "Reusing cached raw datasets")

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
        num_debug_samples = 10  # or 15
        selected_indices = range(min(num_debug_samples, len(train_combined)))
        train_combined = Subset(train_combined, selected_indices)

    train_loader = DataLoader(
        train_combined,
        batch_size=dataloader_cfg.batch_size,
        shuffle=True,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
        drop_last=False,
    )

    # TODO this set has more training data then the other branch!
    # 6) Build a small “validation” split out of **private_ds** (90% train / 10% val)
    # val_size = int(0.1 * len(private_ds))
    # if val_size > 0:
    #     train_sub, val_sub = random_split(private_ds, [len(private_ds) - val_size, val_size])
    # else:
    #     train_sub, val_sub = private_ds, private_ds

    # val_loader = DataLoader(
    #     val_sub,
    #     batch_size=dataloader_cfg.batch_size,
    #     shuffle=False,
    #     num_workers=dataloader_cfg.num_workers,
    #     pin_memory=dataloader_cfg.pin_memory,
    # )
    

    # 7) Dirichlet‐partition full_test → client_test_idx
    test_labels = full_test.targets.tolist()
    test_idcs_all = dirichlet_partition(test_labels, cfg.num_clients, num_classes, cfg.alpha)
    client_test_idx = test_idcs_all[cid]

    test_ds = Subset(full_test, client_test_idx)

    if debug:
        num_debug_samples = 10
        selected_indices = range(min(num_debug_samples, len(test_ds)))
        test_ds = Subset(test_ds, selected_indices)

    test_loader = DataLoader(
        test_ds,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        num_workers=dataloader_cfg.num_workers,
        pin_memory=dataloader_cfg.pin_memory,
    )

    log(INFO, f"[Client {cid}] private={len(private_idx)} + shared={per_client_G} → train={len(train_combined)} | test={len(test_ds)}")
    return train_loader, test_loader