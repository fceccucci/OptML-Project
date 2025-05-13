import hydra, torch, logging
from omegaconf import DictConfig, OmegaConf

from dataset_factory import build_dataloaders
from model_factory import build_model
from algorithm_factory import build_server

import torchvision, torch

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # 1. Build components ------------------------------------------------------
    trainloaders, valloaders = build_dataloaders(cfg.dataset)
    model_fn = lambda: build_model(cfg.model)     # Flower expects a *callable*
    server = build_server(cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task)

    # 2. Kick-off federated training ------------------------------------------
    server.fit()                                  # single-process (Flower “virtual clients”)

    # 3. Persist final global model -------------------------------------------
    torch.save(server.global_model.state_dict(), "global.pt")

if __name__ == "__main__":
    main()
