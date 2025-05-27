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
        return FedYogi(
            eta=cfg.algorithm.eta,
            beta_1=cfg.algorithm.beta1,
            beta_2=cfg.algorithm.beta2,
            tau=cfg.algorithm.tau,
            **common_kwargs,
        )
    if name == "fedadam":
        return FedAdam(
            eta        = cfg.algorithm.eta,        # final LR, e.g. 0.10
            eta_late   = cfg.algorithm.eta,        # keep the same later
            eta_warmup = cfg.algorithm.eta_warmup, 
            beta_1     = cfg.algorithm.beta1,
            beta_2     = cfg.algorithm.beta2,
            tau        = cfg.algorithm.tau,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")