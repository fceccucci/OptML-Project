from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedYogi, FedAvgM

def get_fl_algo(cfg, global_model_init, evaluate_global, standard_aggregate):
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
    elif name == "fedadam":
        return FedAdam(eta=cfg.algorithm.eta, **common_kwargs)
    elif name == "fedyogi":
        return FedYogi(eta=cfg.algorithm.eta, **common_kwargs)
    elif name == "fedavgm":
        return FedAvgM(**common_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {name}")