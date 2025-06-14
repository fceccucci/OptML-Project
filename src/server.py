"""
Defines the CustomServer class, a Flower Server subclass that logs federated learning
metrics to Weights & Biases (wandb) during training and evaluation.
"""
import os, sys
sys.path.insert(0, os.getcwd())

from flwr.server import Server
import timeit
from logging import INFO
from typing import Optional
from flwr.server.history import History
from flwr.common.logger import log

from src.history import WAndBHistory

class CustomServer(Server):
    """
    A custom Flower Server that logs training and evaluation metrics to Weights & Biases (wandb).

    This server overrides the default fit loop to:
      - Log initial evaluation metrics.
      - Log metrics after each federated round (both centralized and distributed).
      - Track elapsed training time.
      - Use a custom History object (WAndBHistory) for wandb integration.
    """

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """
        Run federated averaging for a number of rounds, logging metrics to wandb.

        Args:
            num_rounds (int): Number of federated rounds to run.
            timeout (Optional[float]): Timeout for each round.

        Returns:
            Tuple[History, float]: The training history and total elapsed time.
        """
        history = WAndBHistory()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed