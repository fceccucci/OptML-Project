# Federated Learning with PyTorch Lightning and Flower

This project simulates **Federated Learning (FL)** using the [Flower](https://flower.dev/) framework and [PyTorch Lightning](https://www.pytorchlightning.ai/). It supports multiple FL strategies (e.g., FedAvg, FedYogi, FedAdam, FedProx, FedAvgM) and allows experimentation with non-IID data distributions via configurable YAML files for models, datasets, and algorithms.

---

##  What is Federated Learning?

Federated Learning is a decentralized machine learning paradigm where clients (e.g., devices or data silos) train models locally on their private data. Only model updates (not raw data) are sent to a central server for aggregation. This enhances **data privacy**, **scalability**, and **compliance with data regulations**.

This project simulates that process using Flower's simulation engine and PyTorch Lightning's training loop.

---

##  Project Structure

```
├── src/
│   ├── client_app.py          # Entry point for client simulation
│   ├── server_app.py          # Entry point for the FL server
│   ├── model.py               # Model definitions (e.g., CNN)
│   ├── dataset_factory.py     # Dataset loading and non-IID partitioning
│   ├── straregy_factory.py    # FL strategy instantiation
│   ├── utils.py               # Utilities (e.g., seeding, logging)
│   ├── history.py             # Logging metrics across rounds
│   ├── debug.py               # Debugging utilities
├── conf/
│   ├── *.yaml                 # Hydra configs for models, datasets, algorithms
├── pyproject.toml             # Dependency and configuration specification
├── setup.py                   # (Optional) Package installation script
├── main.py                    # Main script of the project
```

---

##  Dependencies

Install all required dependencies using:

```bash
pip install .
```

Or manually:

```bash
pip install torch==2.6.0 torchvision==0.21.0 pytorch-lightning==2.4.0 \
hydra-core==1.3.2 omegaconf>=2.3.0 flwr==1.18.0 fedlab==1.3.0 \
matplotlib==3.10.3 numpy==2.0.1 flwr[simulation]>=1.18.0 \
flwr-datasets[vision]>=0.5.0 wandb tensorboard>=2.17.0,<3.0.0 \
hydra-joblib-launcher>=1.2.0
```

**Recommended Python version:** ≥ 3.11

---

##  Running Experiments

All training is launched via `main.py` using Hydra config files in the `conf/` directory.

###  Single Run:

```bash
python main.py --config-name=mnist_cnn_debug
```

### Multi-run Sweep:

```bash
python main.py -m --config-name=mnist_sweep
```

---

##  Available Experiments (Hydra Configs)

| Config File                  | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `mnist_cnn_debug.yaml`       | Minimal debug run with 2 clients and few rounds |
| `mnist_cnn_small_local.yaml` | Lightweight local MNIST experiment              |
| `mnist_cnn_server.yaml`      | Full MNIST federated run                        |
| `mnist_cnn_shared.yaml`      | FL with shared public data                      |
| `mnist_sweep.yaml`           | Sweeps over FL strategies and hyperparameters   |

---

## Supported FL Algorithms

| Algorithm       | Description                               |
| --------------- | ----------------------------------------- |
| `fedavg`        | Federated Averaging                       |
| `fedavgm`       | FedAvg with Momentum                      |
| `fedyogi`       | Yogi-style adaptive optimization          |
| `fedadam`       | Adam-based federated optimizer            |
| `fedprox`       | FedAvg with proximal regularization       |
| `fedavg_shared` | FedAvg with additional shared public data |

Configured in YAML via `algorithm.name`.

---

##  Performance & GPU

* Set available GPU via `backend_config.client_resources.num_gpus`.
* Enable mixed-precision via `trainer.precision` (e.g., `bf16-mixed`, `16-mixed`).
* Shared public datasets (for semi-supervised FL) are handled via `shared_data` config blocks.

---

##  Troubleshooting

* Check CUDA availability:

  ```python
  import torch; print(torch.cuda.is_available())
  ```
* Set `debug: True` in configs for verbose logging.
* For performance bottlenecks:

  * Reduce `num_clients`
  * Use smaller datasets (e.g., `FakeData`)
  * Lower `local_epochs` or enable CPU-only simulation
