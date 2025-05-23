"""
utils.py

Utility functions for federated learning experiments, including:
- Setting random seeds for reproducibility
- Generating filenames for saved models based on experiment configuration
- Evaluating trained models on the MNIST test set

These utilities are designed to be imported and used throughout the project.
"""

import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms

def set_seed(seed=42):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The random seed to use (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_filename_from_cfg(cfg, num_rounds):
    """
    Automatically generate a filename for the saved model based on config parameters.
    Includes model, dataset, algorithm, partition type, and any relevant hyperparameters.
    """
    model_name = cfg.model.arch
    dataset_name = cfg.dataset.name
    algorithm_name = cfg.algorithm.name

    # Partition type (iid or non-iid/alpha)
    if hasattr(cfg.dataset, "partition") and cfg.dataset.partition is not None:
        partition_type = str(cfg.dataset.partition)
    elif hasattr(cfg.dataset, "alpha") and cfg.dataset.alpha is not None:
        partition_type = f"alpha{str(cfg.dataset.alpha).replace('.', '')}"
    else:
        partition_type = "iid"

    # Add any other relevant hyperparameters from cfg (customize as needed)
    extra_params = []
    # Add client learning rate if present
    if hasattr(cfg.algorithm, "lr"):
        extra_params.append(f"clientLR{cfg.algorithm.lr}")
    # Add client fraction if present
    if hasattr(cfg.algorithm, "client_fraction"):
        extra_params.append(f"clientFrac{cfg.algorithm.client_fraction}")
    # Add batch size if present
    if hasattr(cfg.dataset, "batch_size"):
        extra_params.append(f"bs{cfg.dataset.batch_size}")

    # Compose filename
    param_str = "_".join([partition_type] + extra_params) if extra_params else partition_type
    filename = f"TrainedModels/{model_name}_{dataset_name}_{algorithm_name}_{param_str}_rounds{num_rounds}_global.pt"
    return filename

def evaluate_on_mnist_test(model_cfg, model_weights_path, batch_size=64, verbose=True):
    """
    Loads the MNIST test set, evaluates the given model on it, and prints accuracy.

    Args:
        model_cfg: The model config (e.g., cfg.model from Hydra).
        model_weights_path: Path to the saved model weights.
        batch_size: Batch size for DataLoader.
        verbose: If True, prints the accuracy.

    Returns:
        accuracy (float): Test accuracy in percent.
    """
    from src.model_factory import build_model

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
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

    accuracy = 100 * correct / total
    if verbose:
        print(f"Global model accuracy on MNIST test set: {accuracy:.2f}%")
    return accuracy


def evaluate_on_cifar10_test(model_cfg, model_weights_path, batch_size=64, verbose=True):
    """
    Loads the CIFAR-10 test set, evaluates the given model on it, and prints accuracy.

    Args:
        model_cfg: The model config (e.g., cfg.model from Hydra).
        model_weights_path: Path to the saved model weights.
        batch_size: Batch size for DataLoader.
        verbose: If True, prints the accuracy.

    Returns:
        accuracy (float): Test accuracy in percent.
    """
    from src.model_factory import build_model

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
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

    accuracy = 100 * correct / total
    if verbose:
        print(f"Global model accuracy on CIFAR-10 test set: {accuracy:.2f}%")
    return accuracy