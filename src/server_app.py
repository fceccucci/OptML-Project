"""pytorchlightning_example: A Flower / PyTorch Lightning app."""
import pytorch_lightning as pl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.server.strategy import FedAvg
from omegaconf import OmegaConf
import torch
from src.utils import get_parameters, set_parameters, standard_aggregate, load_data_test_data_loader, load_data
from src.model import SmallCNN

from src.server import CustomServer
from src.globals import CONFIG_FILE




def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    if 'config' in context.run_config:
        cfg = context.run_config['config']
    # TODO this is outdated
    # config_name = f"{context.run_config['config-name']}" if context.run_config else CONFIG_FILE
    # config_path = f"conf/{config_name}.yaml"
    # cfg = OmegaConf.load(config_path)

    load_data(1,10, cfg)
    # Convert model parameters to flwr.common.Parameters
    global_model = SmallCNN(lr=cfg.algorithm.lr)
    ndarrays = get_parameters(global_model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # common_kwargs = dict(
    #     fraction_fit=algo_cfg.client_fraction,
    #     min_fit_clieserver_fnnts=num_clients,
    #     min_available_clients=num_clients,
    #     on_fit_config_fn=lambda _rnd: {"local_epochs": local_epochs, "lr": lr},
    #     evaluate_fn=evaluate_global,
    #     fit_metrics_aggregation_fn=logging_without_aggregate("train"),
    #     evaluate_metrics_aggregation_fn=logging_without_aggregate("test"),
    # )
    
    test_loader = load_data_test_data_loader(cfg)

    # TODO No global saving possible
    # best = { 'acc': 0, 'parameter': None }
    def evaluate_global(server_rounds, parameters, config):
        set_parameters(global_model, parameters)
        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(global_model, test_loader, verbose=False)
        loss = results[0]["test_loss"]
        acc = results[0]["test_acc"]
        # if acc > best_acc:
        #     best_acc = acc
        #     best_parameter = parameters
        if server_rounds >= (cfg.task.num_of_rounds - 1):
            torch.save(parameters, "best_model.pt")
            # torch.save(best_parameter, "best_model.pt")
        return loss, results[0]


    # TODO build strategy factory
    strategy = FedAvg(
        min_fit_clients=cfg.dataset.num_clients,
        min_available_clients=cfg.dataset.num_clients,
        fraction_fit=cfg.algorithm.client_fraction,
        fraction_evaluate=cfg.algorithm.client_fraction,
        initial_parameters=global_model_init,
        evaluate_fn=evaluate_global,
        evaluate_metrics_aggregation_fn=standard_aggregate,
        fit_metrics_aggregation_fn=standard_aggregate, 
    )

    # Construct ServerConfig
    num_rounds = cfg.task.num_of_rounds
    # TODO get out of config
    config = ServerConfig(num_rounds=num_rounds)
    # return ServerAppComponents(strategy=strategy, config=config)
    return ServerAppComponents(server=CustomServer(strategy=strategy, client_manager=SimpleClientManager()), config=config)


app = ServerApp(server_fn=server_fn)
