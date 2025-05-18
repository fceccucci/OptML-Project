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
from flwr.common import Context, parameters_to_ndarrays

# --------------------------------------------------------------------------- #
#  plain PyTorch helpers                                                      #
# --------------------------------------------------------------------------- #
def _train(model: nn.Module, loader: DataLoader, loss_fn, device, lr: float, epochs: int):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            print(f"[DEBUG] Epoch {epoch+1}/{epochs}, Loss: {total_loss/batch_count:.4f}, Device: {next(model.parameters()).device}")
        else:
            print(f"[WARNING] Client has no data in epoch {epoch+1}.")

def _evaluate(model: nn.Module, loader: DataLoader, loss_fn, device):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"[DEBUG] Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# --------------------------------------------------------------------------- #
#  Custom strategy to save final parameters                                   #
# --------------------------------------------------------------------------- #
class SaveLastParametersFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated[0]
        return aggregated

class SaveLastParametersFedProx(fl.server.strategy.FedProx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated[0]
        return aggregated

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
        
        def to_client(self):
            return fl.client.NumPyClient.to_client(self)

    # -- Strategy -------------------------------------------------------------
    name = algo_cfg.name.lower()
    common_kwargs = dict(
        fraction_fit=algo_cfg.client_fraction,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda _rnd: {"local_epochs": local_epochs, "lr": lr},
    )

    if name == "fedavg":
        strategy = SaveLastParametersFedAvg(**common_kwargs)
    elif name == "fedprox":
        strategy = SaveLastParametersFedProx(
            proximal_mu=getattr(algo_cfg, "mu", 0.0), **common_kwargs
        )
    else:
        raise ValueError(f"Unsupported FL algorithm: {algo_cfg.name}")

    # -- Simple wrapper so main.py just calls .fit() --------------------------
    class _Server:
        def __init__(self):
            self._global_model = model_fn().to(device)

        def fit(self, num_rounds: int = getattr(algo_cfg, "rounds", 50), client_resources=None):
            if client_resources is None:
                client_resources = {"num_cpus": 1}
            def client_fn(context: Context):
                cid = int(context.node_config["partition-id"])
                return _Client(cid).to_client()

            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                client_resources=client_resources,
            )
            # After simulation, get the latest parameters from the strategy
            parameters = strategy.final_parameters
            ndarrays = parameters_to_ndarrays(parameters)
            state_dict = dict(zip(self._global_model.state_dict().keys(),
                                  [torch.tensor(p) for p in ndarrays]))
            self._global_model.load_state_dict(state_dict, strict=True)

        @property
        def global_model(self):
            return self._global_model

    return _Server()