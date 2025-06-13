from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedYogi, FedAvgM

def get_fl_algo(cfg, global_model_init, evaluate_global, standard_aggregate):
    """
    Instantiate and return a federated learning strategy based on the configuration.

    This function selects and configures a Flower FL strategy (FedAvg, FedProx, FedAdam,
    FedYogi, or FedAvgM) according to the algorithm name and hyperparameters in the config.

    Args:
        cfg: Hydra config object containing algorithm and dataset parameters.
        global_model_init: Initial model parameters (as returned by Flower).
        evaluate_global: Function to evaluate the global model.
        standard_aggregate: Function to aggregate metrics across clients.

    Returns:
        A Flower strategy instance configured for the experiment.

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    name = cfg.algorithm.name.lower()
    common_kwargs = dict(
        min_fit_clients=cfg.dataset.num_clients,
        min_available_clients=cfg.dataset.num_clients,
        fraction_fit=cfg.algorithm.client_fraction,
        fraction_evaluate=cfg.algorithm.client_fraction,
        initial_parameters=global_model_init,
        evaluate_fn=evaluate_global,
        evaluate_metrics_aggregation_fn=standard_aggregate,
        fit_metrics_aggregation_fn=standard_aggregate,
    )
    if name == "fedavg":
        return FedAvg(**common_kwargs)
    elif name == "fedprox":
        return FedProx(proximal_mu=cfg.algorithm.mu, **common_kwargs)
    elif name == "fedyogi":
        return FedYogi(
            eta = cfg.algorithm.eta_yogi,
            beta_1 = cfg.algorithm.beta1,
            beta_2 = cfg.algorithm.beta2,
            tau = cfg.algorithm.tau,
            **common_kwargs,
        )
    elif name == "fedadam":
        return FedAdam(
            eta = cfg.algorithm.eta,
            beta_1 = cfg.algorithm.beta1,
            beta_2 = cfg.algorithm.beta2,
            tau = cfg.algorithm.tau,
            **common_kwargs,
        )
    elif name == "fedavgm":
        return FedAvgM(
             server_momentum=cfg.algorithm.momentum,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")