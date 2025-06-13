import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy



class SmallCNN(pl.LightningModule):
    """
    A PyTorch Lightning wrapper for a small convolutional neural network (CNN).

    This class defines a simple CNN architecture for image classification tasks,
    along with training, validation, and test steps using PyTorch Lightning's
    modular API. It also sets up metric collections for accuracy tracking.

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels (e.g., 1 for grayscale).
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1, lr: float = 1e-3):
        """
        Initialize the SmallCNN model, loss function, and metrics.

        Args:
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            lr (float): Learning rate.
        """
        super().__init__()
        # Save hyperparameters (num_classes will be used by metrics)
        self.save_hyperparameters()

        self.lr = lr

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics: use multiclass accuracy with explicit num_classes
        n_classes = self.hparams.num_classes
        self.train_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})
        self.val_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})
        self.test_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        # return self.classifier(self.features(x))
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        """
        Training logic for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # update metric
        self.train_metrics.update(logits, y)
        # log loss per batch
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        """
        Compute and log training metrics at the end of the epoch, then reset metrics.
        """
        metrics = self.train_metrics.compute()
        self.log('train_acc', metrics['accuracy'], prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation logic for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_metrics.update(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics at epoch end, then reset
        """
        metrics = self.val_metrics.compute()
        self.log('val_acc', metrics['accuracy'], prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Test logic for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_metrics.update(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        """
        Compute and log test metrics at epoch end, then reset
        """
        metrics = self.test_metrics.compute()
        self.log('test_acc', metrics['accuracy'], prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        return torch.optim.Adam(self.parameters(),self.lr)
      
