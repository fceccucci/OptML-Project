import hydra, torch, logging
from omegaconf import DictConfig, OmegaConf
from src.dataset_factory import build_dataloaders
from src.model_factory import build_model
from src.algorithm_factory import build_server
from src.utils import set_seed, get_filename_from_cfg, evaluate_on_mnist_test
import torch
import numpy as np
import warnings

# Set random seed for reproducibility
set_seed(42)  

# Suppress Python warnings (e.g., DeprecationWarning, UserWarning)
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)   # Hide Ray warnings (if any)
logging.getLogger("py.warnings").setLevel(logging.ERROR)  # Hide warnings from warnings module

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="mnist_cnn_fedprox_iid.yaml", version_base="1.3")
def main(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True

    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # 1. Build components ------------------------------------------------------
    trainloaders, valloaders = build_dataloaders(cfg.dataset)
    model_fn = lambda: build_model(cfg.model)     # Flower expects a *callable*
    server = build_server(cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task)

    # --- GPU splitting for clients ---
    num_clients = len(trainloaders)
    num_gpus = torch.cuda.device_count()
    client_resources = {"num_cpus": 1}
    if num_gpus > 0:
        client_resources["num_gpus"] = num_gpus / num_clients  # e.g., 1 GPU / 5 clients = 0.2

    server = build_server(cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task)

    # 2. Kick-off federated training ------------------------------------------
    num_rounds = 5

    server.fit(num_rounds=num_rounds)

    # 3. Persist final global model -------------------------------------------
    filename = get_filename_from_cfg(cfg=cfg , num_rounds=num_rounds)
    torch.save(server.global_model.state_dict(), filename)
    log.info(f"Saved global model to {filename}")

     # 4. Evaluate global model on the real MNIST test set ---------------------
    evaluate_on_mnist_test(cfg.model, filename)


if __name__ == "__main__":
    main()
