import hydra, torch, logging
from omegaconf import DictConfig, OmegaConf
from src.dataset_factory import build_dataloaders
from src.model_factory import build_model
from src.algorithm_factory import build_server
from src.utils import set_seed, get_filename_from_cfg
from src.utils import evaluate_on_mnist_test, evaluate_on_cifar10_test
import warnings

set_seed(42)
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="cifar_resnet18_iid.yaml", version_base="1.3")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    log.info(OmegaConf.to_yaml(cfg, resolve=True))

    # 1. Build components
    trainloaders, valloaders = build_dataloaders(cfg.dataset)
    model_fn = lambda: build_model(cfg.model)
    server = build_server(cfg.algorithm, model_fn, trainloaders, valloaders, cfg.task)

    # 2. Set client resources
    num_clients = len(trainloaders)
    num_gpus = torch.cuda.device_count()
    client_resources = {"num_cpus": 1}
    if num_gpus > 0:
        client_resources["num_gpus"] = num_gpus / num_clients

    # 3. Federated training
    num_rounds = getattr(cfg.algorithm, "rounds", 5)
    server.fit(num_rounds=num_rounds, client_resources=client_resources)

    # 4. Save model
    filename = get_filename_from_cfg(cfg=cfg, num_rounds=num_rounds)
    torch.save(server.global_model.state_dict(), filename)
    log.info(f"Saved global model to {filename}")

    # 5. Evaluate
     # 5. Evaluate on correct test set
    dataset_name = cfg.dataset.name.lower()
    if dataset_name == "mnist":
        evaluate_on_mnist_test(cfg.model, filename)
    elif dataset_name == "cifar10":
        evaluate_on_cifar10_test(cfg.model, filename)
    else:
        log.warning(f"No evaluation implemented for dataset: {dataset_name}")


if __name__ == "__main__":
    main()