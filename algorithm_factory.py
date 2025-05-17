"""
algorithm_factory.py
~~~~~~~~~~~~~~~~~~~~
Creates a *server* object with a `.fit()` method that launches a Flower
single-machine simulation.  Currently supports FedAvg and FedProx.  Extend
the strategy block to add more FL algorithms.
"""

from typing import Any, Callable, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from hydra.utils import instantiate


# --------------------------------------------------------------------------- #
#  plain PyTorch helpers                                                      #
# --------------------------------------------------------------------------- #
def _train(model: nn.Module, loader: DataLoader, loss_fn, device, lr: float, epochs: int):
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()


def _evaluate(model: nn.Module, loader: DataLoader, loss_fn, device):
    model.to(device)
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total / n, correct / n


# --------------------------------------------------------------------------- #
#  PUBLIC API                                                                 #
# --------------------------------------------------------------------------- #
def build_server(
    algo_cfg: Any,
    model_fn: Callable[[], nn.Module],
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    task_cfg: Any,
):
    """
    Returns
    -------
    server : object
        Call `server.fit()` to kick off federated training.
    """
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients  = len(trainloaders)
    loss_fn      = instantiate(task_cfg.loss) if hasattr(task_cfg, "loss") else nn.CrossEntropyLoss()
    local_epochs = algo_cfg.local_epochs
    lr           = algo_cfg.lr

    # -- Flower client --------------------------------------------------------
    class _Client(fl.client.NumPyClient):
        def __init__(self, cid: str):
            self.cid   = int(cid)
            self.model = model_fn().to(device)

        # Flower <> Torch parameter helpers ----------------------------------
        def get_parameters(self, config):
            return [val.cpu().numpy() for val in self.model.state_dict().values()]

        def set_parameters(self, params):
            state_dict = dict(zip(self.model.state_dict().keys(),
                                  [torch.tensor(p) for p in params]))
            self.model.load_state_dict(state_dict, strict=True)

        # Flower hooks --------------------------------------------------------
        def fit(self, params, config):
            self.set_parameters(params)
            _train(self.model, trainloaders[self.cid], loss_fn, device, lr, local_epochs)
            return self.get_parameters(None), len(trainloaders[self.cid].dataset), {}

        def evaluate(self, params, config):
            self.set_parameters(params)
            loss, acc = _evaluate(self.model, valloaders[self.cid], loss_fn, device)
            return float(loss), len(valloaders[self.cid].dataset), {"accuracy": acc}

    # -- Strategy -------------------------------------------------------------
    name = algo_cfg.name.lower()
    common_kwargs = dict(
        fraction_fit=algo_cfg.client_fraction,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda _rnd: {"local_epochs": local_epochs, "lr": lr},
    )

    if name == "fedavg":
        strategy = fl.server.strategy.FedAvg(**common_kwargs)
    elif name == "fedprox":
        strategy = fl.server.strategy.FedProx(
            proximal_mu=getattr(algo_cfg, "mu", 0.0), **common_kwargs
        )
    else:
        raise ValueError(f"Unsupported FL algorithm: {algo_cfg.name}")

    # -- Simple wrapper so main.py just calls .fit() --------------------------
        
    class _Server:
        def __init__(self):
            self._global_model = model_fn().to(device)

        def fit(self, num_rounds: int = getattr(algo_cfg, "rounds", 50)):
            fl.simulation.start_simulation(
                client_fn=lambda cid: _Client(cid),
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                client_resources={"num_cpus": 1, "num_gpus": 1.0},  # <--- Add this line
            )

        @property
        def global_model(self):
            return self._global_model


    return _Server()
