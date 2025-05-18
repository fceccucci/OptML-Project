"""
model_factory.py
~~~~~~~~~~~~~~~~
Return a freshly initialised torch.nn.Module according to the Hydra
`conf/model/*.yaml` node that main.py passes in.

Supported architectures out-of-the-box
--------------------------------------
* cnn        – a lightweight 3-conv CNN for CIFAR-10 / MNIST
* resnet18   – torchvision.resnet18, with optional pretrained weights

Add more by extending the `if / elif` ladder inside `build_model`.
"""

from typing import Any
import torch.nn as nn
import torchvision.models as tv


# --------------------------------------------------------------------------- #
#  Tiny baseline CNN (32-64-128 channels)                                     #
# --------------------------------------------------------------------------- #
class _SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))



# --------------------------------------------------------------------------- #
#  PUBLIC FACTORY                                                             #
# --------------------------------------------------------------------------- #
def build_model(cfg: Any) -> nn.Module:
    """
    Parameters
    ----------
    cfg : DictConfig node defined in `conf/model/<name>.yaml`
        Required keys: arch, num_classes
        Optional:     pretrained (bool)  – only for torchvision models

    Returns
    -------
    torch.nn.Module
    """
    arch = cfg.arch.lower()

    if arch == "cnn":
        # Use 3 channels for MNIST (since you use Grayscale(num_output_channels=3))
        in_channels = 3
        return _SmallCNN(cfg.num_classes, in_channels=in_channels)


    if arch == "resnet18":
        weights = None
        if getattr(cfg, "pretrained", False):
            # Use the recommended weights enum for torchvision >= 0.13
            try:
                weights = tv.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = "IMAGENET1K_V1"  # fallback for older versions
        return tv.resnet18(
            weights=weights,
            num_classes=cfg.num_classes,
        )

    # ---- add new architectures here ---------------------------------------
    # elif arch == "mobilenetv3":
    #     return tv.mobilenet_v3_small(num_classes=cfg.num_classes)
    # -----------------------------------------------------------------------

    raise ValueError(f"Unknown architecture: {cfg.arch}")
