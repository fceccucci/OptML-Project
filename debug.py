# main.py
import hydra
from omegaconf import DictConfig, OmegaConf

from src.server_app import server_fn 
from src.client_app import Client, client_fn
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context
from src.utils import flatten_dict
import wandb

@hydra.main(config_path="conf", config_name="mnist_cnn_debug")
def main(cfg: DictConfig) -> None:
    # Convert the entire Hydra config into a plain dict (so it's JSON-serializable)
    hydra_config = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project="mnist_cnn", config=flatten_dict(cfg))

    # Wrap the client_fn to inject our Hydra config into the run context
    def hydra_client_fn(context: Context) -> Client:
        context.run_config["config"] = cfg
        return client_fn(context)

    # Wrap the server_fn similarly
    def hydra_server_fn(context: Context) -> ServerAppComponents:
        context.run_config["config"] = cfg
        return server_fn(context)

    # Build Flower apps
    server_app = ServerApp(server_fn=hydra_server_fn)
    client_app = ClientApp(client_fn=hydra_client_fn)

    backend_conf   = hydra_config['backend_config']

    # Launch simulation, pulling any Flower‐specific params from cfg if you like
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.dataset.num_clients,
        backend_config=backend_conf
    )

if __name__ == "__main__":
    main()
