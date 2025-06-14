"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

from omegaconf import OmegaConf
import pytorch_lightning as pl
# from datasets.utils.logging import disable_progress_bar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from src.model import SmallCNN
from src.dataset_factory import load_dataset, build_dataloaders
from src.globals import CONFIG_FILE
from src.utils import (
    get_parameters,
    set_parameters,
    load_data,
    set_seed,
    get_best_device,
)


class FlowerClient(NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, max_epochs, cfg):
        # TODO make this changeble
        self.model = SmallCNN(lr=cfg.algorithm.lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.cfg = cfg

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_parameters(self.model, parameters)

        trainer = pl.Trainer(max_epochs=self.max_epochs,
                            accelerator=get_best_device(),
                            precision=self.cfg.trainer.precision,
                            enable_progress_bar=False,
                            enable_checkpointing=False,  
                            gradient_clip_val=1.0,
                       )
        # TODO val loader not needed in 1 to 5 epochs!
        trainer.fit(self.model, train_dataloaders=self.train_loader)
        # trainer.fit(self.model, self.train_loader, self.val_loader)
        metrics  = {k: v.detach().cpu().item()       # convert Tensor → Python float
            for k, v in trainer.callback_metrics.items()}
        # TODO how to get the metrics out of this?!
        return get_parameters(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)

        trainer = pl.Trainer(enable_progress_bar=False, accelerator=get_best_device(), enable_checkpointing=False,  )
        results = trainer.test(self.model, self.test_loader, verbose=False)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), results[0]


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    cfg = context.cfg
    set_seed(42)
    # TODO this is outdated
    # config_name = f"{context.run_config['config-name']}" if context.run_config else CONFIG_FILE
    # config_path = f"conf/{config_name}.yaml"
    # cfg = OmegaConf.load(config_path)


    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"] #TODO maybe use this value instead of the one in the config
    assert num_partitions == cfg.dataset.num_clients
    
    # train_loader, val_loader, test_loader = load_data(partition_id, num_partitions, cfg)

    # TODO the actual dataset has a lot of buffering problems
    train_loader, val_loader, test_loader= build_dataloaders(cfg.dataset, cfg.dataloader, cfg.debug, partition_id)
    
    # Read run_config to fetch hyperparameters relevant to this run
    # max_epochs = context.run_config["max-epochs"]
    max_epochs = cfg.algorithm.local_epochs
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs, cfg).to_client()

app = ClientApp(client_fn=client_fn)
