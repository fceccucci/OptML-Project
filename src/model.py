import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy



class SmallCNN(pl.LightningModule):
    """
    A PyTorch Lightning wrapper for the SmallCNN model with proper MetricCollection usage.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 1, lr: float = 1e-3):
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
  

        # # Feature extractor
        # self.features = nn.Sequential(
        #     # nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2),
        #     # nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        
        # )
        # # Classifier head
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(7 * 7 * 128, 256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(256, num_classes),
        # )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics: use multiclass accuracy with explicit num_classes
        n_classes = self.hparams.num_classes
        self.train_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})
        self.val_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})
        self.test_metrics = MetricCollection({'accuracy': Accuracy(task="multiclass", num_classes=n_classes)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # return self.classifier(self.features(x))
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        """Training logic: update metrics each batch"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # update metric
        self.train_metrics.update(logits, y)
        # log loss per batch
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        """Compute and log training metrics at epoch end, then reset"""
        metrics = self.train_metrics.compute()
        self.log('train_acc', metrics['accuracy'], prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """Validation logic: update metrics each batch"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_metrics.update(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end, then reset"""
        metrics = self.val_metrics.compute()
        self.log('val_acc', metrics['accuracy'], prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Test logic: update metrics each batch"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_metrics.update(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        """Compute and log test metrics at epoch end, then reset"""
        metrics = self.test_metrics.compute()
        self.log('test_acc', metrics['accuracy'], prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Optimizer configuration"""
        return torch.optim.Adam(self.parameters(),self.lr)
        # return torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.lr,
        #     momentum=0.9
        # )
