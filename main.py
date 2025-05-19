import hydra, torch, logging
from omegaconf import DictConfig, OmegaConf
from dataset_factory import build_dataloaders
from model_factory import build_model
from algorithm_factory import build_server
import torchvision, torch
import random
import numpy as np
import warnings

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

log = logging.getLogger(__name__)


warnings.filterwarnings("ignore")



@hydra.main(config_path="conf", config_name="mnist_cnn.yaml", version_base="1.3")
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
    model_name = cfg.model.arch
    dataset_name = cfg.dataset.name
    algorithm_name = cfg.algorithm.name
    filename = f"TrainedModels/{model_name}_{dataset_name}_{algorithm_name}_alpha0005_rounds{num_rounds}_global.pt"
    torch.save(server.global_model.state_dict(), filename)
    log.info(f"Saved global model to {filename}")

     # 4. Evaluate global model on the real MNIST test set ---------------------
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Global model accuracy on MNIST test set: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
