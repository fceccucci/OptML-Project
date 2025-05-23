from flwr.server.history import History
import wandb
from flwr.common.typing import Scalar


class WAndBHistory(History):
    """
    History subclass that logs metrics to Weights & Biases (wandb).
    Initializes a wandb run on creation and logs distributed and centralized
    losses and metrics at each round.
    """

    def __init__(self, project: str = "flower", **wandb_init_kwargs) -> None:
        # Initialize base History
        super().__init__()
        # Start a new wandb run
        if wandb.run is not None:
            wandb.init(project=project, **wandb_init_kwargs)

    def add_loss_distributed(self, server_round: int, loss: float) -> None:
        # Log distributed loss
        wandb.log({"distributed/loss": loss}, step=server_round)
        # Call base implementation
        super().add_loss_distributed(server_round, loss)

    def add_loss_centralized(self, server_round: int, loss: float) -> None:
        # Log centralized loss
        wandb.log({"centralized/loss": loss}, step=server_round)
        # Call base implementation
        super().add_loss_centralized(server_round, loss)

    def add_metrics_distributed_fit(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        # Prefix each metric key and log
        log_data = {f"distributed_fit/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_distributed_fit(server_round, metrics)

    def add_metrics_distributed(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        # Prefix each metric key and log
        log_data = {f"distributed_eval/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_distributed(server_round, metrics)

    def add_metrics_centralized(
        self, server_round: int, metrics: dict[str, Scalar]
    ) -> None:
        # Prefix each metric key and log
        log_data = {f"centralized/{key}": value for key, value in metrics.items()}
        wandb.log(log_data, step=server_round)
        super().add_metrics_centralized(server_round, metrics)
