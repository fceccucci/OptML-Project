"""pytorchlightning_example: A Flower / PyTorch Lightning app."""
import pytorch_lightning as pl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.server.strategy import FedAvg
from omegaconf import OmegaConf
import torch
from src.utils import get_parameters, set_parameters
from src.model import SmallCNN
from src.dataset_factory import load_dataset
from src.server import CustomServer




def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    print(context.run_config)
    print(context.run_config['config-name'])
    return
    
    config_path = f"conf/{context.run_config['config-name']}.yaml"
    cfg = OmegaConf.load(config_path)
    #TODO use cfg to create run!
    # Convert model parameters to flwr.common.Parameters
    global_model = SmallCNN()
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
    _, _, test_dataset = load_dataset(cfg.dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    def evaluate_global(server_rounds, parameters, config):
        # print(f"weights:{weights}")
        # print(f"config:{config}")
        # Load weights into a fresh model
        set_parameters(global_model, parameters)
        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(global_model, test_loader)
        loss = results[0]["test_loss"]
        # TODO how to get the metrics out of this?!
        return loss, {}


    # TODO build strategy factory
    # Define strategy
    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        initial_parameters=global_model_init,
        evaluate_fn=evaluate_global,

    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    # TODO get out of config
    config = ServerConfig(num_rounds=num_rounds)
    # return ServerAppComponents(strategy=strategy, config=config)
    return ServerAppComponents(server=CustomServer(strategy=strategy, client_manager=SimpleClientManager()), config=config)


app = ServerApp(server_fn=server_fn)
