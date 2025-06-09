# Federated Learning with PyTorch Lightning and Flower

This project demonstrates how to simulate Federated Learning (FL) using the Flower framework and PyTorch Lightning. It supports multiple FL strategies (FedAvg, FedYogi, FedAdam, FedProx, FedAvgM) with non-IID data distributions using configurable YAML files for dataset, model, and algorithm management.

---

## What is Federated Learning?

Federated Learning is a decentralized machine learning approach where training happens across multiple devices or servers holding local data samples, without exchanging them. This enables privacy preservation and reduces the need for centralized data aggregation. 

Each client trains the model on its own data and sends only the model updates to a central server for aggregation. This project simulates this process using `Flower` and `PyTorch Lightning`.

---

## Project Structure

```text
├── src/
│   ├── client_app.py          # Federated client simulation
│   ├── server_app.py          # Federated server entry point
│   ├── model.py               # CNN and model factory
│   ├── dataset_factory.py     # Dataset loading and partitioning
│   ├── straregy_factory.py    # FL strategy creation
│   ├── utils.py               # Utility functions
│   ├── history.py             # Tracking training/evaluation
│   ├── debug.py               # Debugging tools
├── conf/
│   ├── *.yaml                 # Hydra configs for models, algorithms, datasets
├── pyproject.toml             # Dependency and config specification
├── setup.py                   # Optional setup script
````

---

## Dependencies

All dependencies are listed in `pyproject.toml`. Install them using:

```bash
pip install .
```

Or manually using:

```bash
pip install torch==2.6.0 torchvision==0.21.0 pytorch-lightning==2.4.0 \
hydra-core==1.3.2 omegaconf>=2.3.0 flwr==1.18.0 fedlab==1.3.0 \
matplotlib==3.10.3 numpy==2.0.1 flwr[simulation]>=1.18.0 \
flwr-datasets[vision]>=0.5.0 wandb tensorboard>=2.17.0,<3.0.0 \
hydra-joblib-launcher>=1.2.0
```

Python ≥3.11 recommended.

---

## How to Run

### Run a full FL simulation

Use Flower’s entrypoint:

```bash
python src/server_app.py
```

This launches the server and all clients using the `mnist_cnn_server.yaml` config by default.

### Run a client independently (for debugging or inspection)

```bash
python src/client_app.py
```

You can specify custom configuration by modifying the YAMLs in the `conf/` folder.

---

## Available Experiments

### Configurable with Hydra:

* `mnist_cnn_debug.yaml` – Debug with 2 clients, few rounds
* `mnist_cnn_small_local.yaml` – Small MNIST experiment
* `mnist_cnn_server.yaml` – Full MNIST federated experiment
* `mnist_sweep.yaml` – Sweeps across different FL algorithms and hyperparameters
* `fake_debug.yaml` – Simulates data using `torchvision.datasets.FakeData`

---

## Supported Algorithms

* `fedavg` – Federated Averaging
* `fedyogi` – Adaptive optimization
* `fedadam` – Adam-based FedOpt
* `fedprox` – Robust optimization with proximal terms
* `fedavgm` – FedAvg with momentum

Customize via YAML (`algorithm.name`).

---

## Model & Logging

Models and training logs are saved via PyTorch Lightning loggers (e.g., TensorBoard, W\&B). You can enable W\&B in `pyproject.toml`.

To log runs locally:

```bash
tensorboard --logdir=lightning_logs/
```

---

## Notes

* GPU usage is configured via `backend_config.client_resources.num_gpus`.
* Mixed-precision is enabled via `trainer.precision` (`bf16-mixed`, `16-mixed`).
* Shared data (e.g., for FedAvg+Shared) is handled with `shared_data` config block.

---

## Troubleshooting

* Ensure CUDA is available if using GPU: `torch.cuda.is_available()`
* Use `debug: True` in YAML to enable verbose logging
* For slow performance, reduce `num_clients` or use `FakeData`

---

