# src/dataset_factory.py
from typing import List, Tuple
import os
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
from torchvision import transforms
import random

# Partition helpers
try:
    from fedlab.utils.dataset.functional import hetero_dir_partition as dirichlet_partition
except ImportError:
    raise ImportError("FedLab is required. Install with `pip install fedlab`.")


def build_dataloaders(cfg) -> Tuple[List[DataLoader], List[DataLoader], torch.utils.data.Dataset]:
    # Load full train + test
    root = os.path.expanduser(getattr(cfg, "root", "/tmp/data"))
    batch_size = getattr(cfg, "batch_size", 32)
    num_clients = cfg.num_clients

    tfm = transforms.Compose([transforms.ToTensor()])
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

    # 2) Partition D non-IID via Dirichlet
    D_labels = [labels[i] for i in D_indices]
    client_D_idcs = dirichlet_partition(D_labels, num_clients, num_classes, cfg.partition.alpha)
    client_train_indices = [[D_indices[j] for j in part] for part in client_D_idcs]

    # 3) Build per-client loaders with α-portion of G
    alpha_dist = getattr(cfg.partition, "alpha_dist", 1.0)
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
        print(f"[INFO] Client {cid}: {len(train_indices)} private + {len(shared_indices)} shared = {len(combined_dataset)} total train samples")
        trainloaders.append(DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(Subset(test_ds, dirichlet_partition([lbl for _,lbl in test_ds],  # reuse split
                                                          num_clients, num_classes, cfg.partition.alpha)[cid]),
                                    batch_size=batch_size, shuffle=False, num_workers=2))
        

    return trainloaders, valloaders, G_dataset


# src/algorithm_factory.py (patch build_server to accept warm-up)
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# ... inside build_server signature add initial_parameters=None
# After strategy instantiation:
if initial_parameters is not None:
    # Convert list of ndarrays into Flower Parameters and set strategy
    strategy.initial_parameters = ndarrays_to_parameters(initial_parameters)

# src/main.py (warm-up + FL)
import hydra, torch
from omegaconf import DictConfig
from src.dataset_factory import build_dataloaders
from src.model_factory import build_model
from src.algorithm_factory import build_server
from src.utils import set_seed, evaluate_on_mnist_test

@hydra.main(config_path="conf", config_name="mnist_cnn_shared.yaml", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Build data
    trainloaders, valloaders, G_dataset = build_dataloaders(cfg.dataset)

    # Warm-up on G
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_model = build_model(cfg.model).to(device)
    opt = torch.optim.SGD(warmup_model.parameters(), lr=cfg.algorithm.lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    loaderG = torch.utils.data.DataLoader(G_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    warmup_model.train()
    for _ in range(cfg.algorithm.warmup_epochs):
        for x, y in loaderG:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(warmup_model(x), y).backward()
            opt.step()

    # Extract warm-up weights
    initial_params = [val.cpu().numpy() for val in warmup_model.state_dict().values()]

    # Build FL server with warm-up
    model_fn = lambda: build_model(cfg.model)
    server = build_server(
        cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task,
        initial_parameters=initial_params
    )

    # Run federated training
    server.fit(num_rounds=cfg.algorithm.rounds)

    # Save & evaluate
    filename = get_filename_from_cfg(cfg, cfg.algorithm.rounds)
    torch.save(server.global_model.state_dict(), filename)
    evaluate_on_mnist_test(cfg.model, filename)

if __name__ == "__main__":
    main()
