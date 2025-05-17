import hydra, torch, logging
from omegaconf import DictConfig, OmegaConf

from dataset_factory import build_dataloaders
from model_factory import build_model
from algorithm_factory import build_server

import torchvision, torch

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="cifar_resnet18_iid.yaml", version_base="1.3")
def main(cfg: DictConfig):

    torch.backends.cudnn.benchmark = True

    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # 1. Build components ------------------------------------------------------
    trainloaders, valloaders = build_dataloaders(cfg.dataset)
    model_fn = lambda: build_model(cfg.model)     # Flower expects a *callable*
    server = build_server(cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task)

    # 2. Kick-off federated training ------------------------------------------
    num_rounds = 20
    
    server.fit(num_rounds=num_rounds)

    # 3. Persist final global model -------------------------------------------
    model_name = cfg.model.arch
    dataset_name = cfg.dataset.name
    algorithm_name = cfg.algorithm.name
    filename = f"TrainedModels/{model_name}_{dataset_name}_{algorithm_name}_iid_rounds{num_rounds}_global.pt"
    torch.save(server.global_model.state_dict(), filename)
    log.info(f"Saved global model to {filename}")


if __name__ == "__main__":
    main()
