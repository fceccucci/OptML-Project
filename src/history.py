"""
his module provides a custom Flower History class that logs federated learning
metrics to Weights & Biases (wandb) during training and evaluation.
"""
from flwr.server.history import History
import wandb
from flwr.common.typing import Scalar


class WAndBHistory(History):
    """
    A History subclass that logs metrics to Weights & Biases (wandb).

    This class extends Flower's History to automatically log distributed and centralized
    losses and metrics to wandb at each federated round. It can be used as a drop-in
    replacement for the standard History class in Flower server code.

    Args:
        project (str): The wandb project name.
        **wandb_init_kwargs: Additional keyword arguments for wandb.init().
    """

    def __init__(self, project: str = "flower", **wandb_init_kwargs) -> None:
        """
        Initialize the WAndBHistory object and (optionally) start a wandb run.

        Args:
            project (str): The wandb project name.
            **wandb_init_kwargs: Additional keyword arguments for wandb.init().
        """
        # Initialize base History
        super().__init__()

    def add_loss_distributed(self, server_round: int, loss: float) -> None:
        """
        Log distributed loss to wandb and call the base implementation.

        Args:
            server_round (int): The current federated round.
            loss (float): The distributed loss value.
        """
        # Log distributed loss
        wandb.log({"distributed/loss": loss}, step=server_round)
        # Call base implementation
        super().add_loss_distributed(server_round, loss)

    def add_loss_centralized(self, server_round: int, loss: float) -> None:
        """
        Log centralized loss to wandb and call the base implementation.

        Args:
            server_round (int): The current federated round.
            loss (float): The centralized loss value.
        """
        # Log centralized loss
        wandb.log({"centralized/loss": loss}, step=server_round)
        # Call base implementation
        super().add_loss_centralized(server_round, loss)

    def add_metrics_distributed_fit(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        """
        Log distributed fit metrics to wandb and call the base implementation.

        Args:
            server_round (int): The current federated round.
            metrics (dict): Dictionary of distributed fit metrics.
        """
        # Prefix each metric key and log
        log_data = {f"distributed_fit/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_distributed_fit(server_round, metrics)

    def add_metrics_distributed(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        """
        Log distributed evaluation metrics to wandb and call the base implementation.

        Args:
            server_round (int): The current federated round.
            metrics (dict): Dictionary of distributed evaluation metrics.
        """
        # Prefix each metric key and log
        log_data = {f"distributed_eval/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_distributed(server_round, metrics)

    def add_metrics_centralized(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        """
        Log centralized metrics to wandb and call the base implementation.

        Args:
            server_round (int): The current federated round.
            metrics (dict): Dictionary of centralized metrics.
        """
        # Prefix each metric key and log
        log_data = {f"centralized/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_centralized(server_round, metrics)
