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
    num_rounds = getattr(cfg.algorithm, "rounds", 5)
    server.fit(num_rounds=num_rounds)

    # Save & evaluate
    filename = get_filename_from_cfg(cfg, num_rounds)
    torch.save(server.global_model.state_dict(), filename)
    evaluate_on_mnist_test(cfg.model, filename)

if __name__ == "__main__":
    main()
