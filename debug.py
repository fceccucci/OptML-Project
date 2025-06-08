# main.py
import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset_factory import build_shared_dataset, build_client_loaders
from src.server_app import server_fn 
from src.client_app import Client, client_fn
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents
from flwr.common import Context
from src.utils import flatten_dict
from hydra.core.hydra_config import HydraConfig
import wandb
import os, socket

@hydra.main(version_base="1.1", config_path="conf", config_name="mnist_cnn_debug")
def main(cfg: DictConfig) -> None:
    # Convert the entire Hydra config into a plain dict (so it's JSON-serializable)
    # only set once if not already in the environment:

    run_name = f"{cfg.algorithm.name}_cf_{cfg.algorithm.client_fraction}_le_{cfg.algorithm.local_epochs}_alpha_{cfg.dataset.alpha}_warmup_{cfg.dataset.share_fraction}_epochs_{cfg.algorithm.warmup_epochs}_alpha_dist_{cfg.dataset.alpha_dist}"

    hydra_config = OmegaConf.to_container(cfg, resolve=True)

    hydra_cfg = HydraConfig.get()
    # Pull out the config_name you passed via @hydra.main:
    cfg_name: str = hydra_cfg.job.config_name
    wandb.init(project=cfg_name[:-5]+"_shared", 
               config=flatten_dict(cfg),
               name=run_name, 
               reinit=True,
               group="federated_sweep")


    # Wrap the client_fn to inject our Hydra config into the run context
    def hydra_client_fn(context: Context) -> Client:
        context.cfg = cfg
        return client_fn(context)

    # Wrap the server_fn similarly
    def hydra_server_fn(context: Context) -> ServerAppComponents:
        context.cfg = cfg
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