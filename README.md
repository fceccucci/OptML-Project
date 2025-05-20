# OptML-Project

Federated learning experiments on MNIST and CIFAR-10 using PyTorch and Flower.

Below is a conceptual diagram of the federated learning workflow implemented in this project:

```
                        +---------------------+
                        |   Central Server    |
                        |  (Aggregator Node)  |
                        +----------+----------+
                                   |
                +------------------+------------------+
                |                  |                  |
        +-------v-------+  +-------v-------+  +-------v-------+
        |   Client 1    |  |   Client 2    |  |   Client N    |
        | (Data Owner)  |  | (Data Owner)  |  | (Data Owner)  |
        +-------+-------+  +-------+-------+  +-------+-------+
                |                  |                  |
        [Local Training]   [Local Training]   [Local Training]
                |                  |                  |
                +--------+  +------+-------+  +-------+
                         |  |              |
                  +------v--v--------------v------+
                  |   Model Updates Sent to       |
                  |      Central Server           |
                  +------------------------------+
                                   |
                        +----------v----------+
                        |   Central Server    |
                        |  (Aggregator Node)  |
                        +---------------------+
                                   |
                        [Global Model Update]
                                   |
                        +---------------------+
                        |   Global Model      |
                        +---------------------+

---

## Project Structure

```
.
├── main.py                      # Entry point for training (Hydra-based)
├── README.md                    # Project documentation
├── TrainedModelEval.ipynb       # Jupyter notebook for model evaluation and visualization
├── conf/                        # YAML configuration files for all experiments
│   ├── mnist_cnn.yaml
│   ├── mnist_cnn_iid.yaml
│   ├── mnist_cnn_fedprox.yaml
│   ├── mnist_cnn_fedprox_iid.yaml
│   ├── cifar_resnet18.yaml
│   ├── cifar_resnet18_iid.yaml
├── src/                         # Source code for modular experiment components
│   ├── algorithm_factory.py     # Federated algorithm (FedAvg, FedProx) server logic
│   ├── dataset_factory.py       # Dataset loading and partitioning (IID/Dirichlet)
│   ├── model_factory.py         # Model definitions and instantiation
│   └── utils.py                 # Utilities (seed, evaluation, filename helpers)
├── TrainedModels/               # Saved global models after training
├── data/                        # Downloaded datasets (MNIST, CIFAR-10)
└── outputs/                     # Training logs and outputs (by date/time)
```

---

## Setup

1. **Install dependencies**  
   It is recommended to use a virtual environment (e.g., conda or venv):

   ```sh
   pip install torch torchvision hydra-core flwr fedlab matplotlib numpy
   ```

   **Library Versions Used:**
   - torch: 2.5.1  
   - torchvision: 0.20.1  
   - hydra-core: 1.3.2  
   - flwr: 1.18.0  
   - fedlab: 1.3.0  
   - matplotlib: 3.10.3  
   - numpy: 2.0.1  

2. **(Optional) Install Jupyter for evaluation notebooks**

   ```sh
   pip install notebook
   ```

---

## Training

All training is launched via `main.py` using Hydra configs from the `conf/` folder.

### Example: Train CNN on MNIST (IID)

```sh
python main.py --config-name=mnist_cnn_iid.yaml
```

### Example: Train CNN on MNIST (Dirichlet non-IID)

```sh
python main.py --config-name=mnist_cnn.yaml
```

### Example: Train FedProx on MNIST (IID)

```sh
python main.py --config-name=mnist_cnn_fedprox_iid.yaml
```

### Example: Train ResNet18 on CIFAR-10 (IID)

```sh
python main.py --config-name=cifar_resnet18_iid.yaml
```

### Example: Train ResNet18 on CIFAR-10 (Dirichlet non-IID)

```sh
python main.py --config-name=cifar_resnet18.yaml
```

- Trained global models are saved in the `TrainedModels/` directory.
- All experiment settings (dataset, model, algorithm, etc.) are controlled via YAML files in the `conf/` directory.  
  You can create new configs by copying and modifying existing ones.

---

## Evaluation

Use the provided Jupyter notebook [`TrainedModelEval.ipynb`](TrainedModelEval.ipynb) to evaluate and visualize the trained models.

1. Open the notebook in VS Code or Jupyter.
2. Adjust the `model_path` in the relevant cells to point to your trained model file in `TrainedModels/`.
3. Run the notebook cells to compute accuracy and visualize predictions.

---

## Configuration

- The code supports both IID and Dirichlet non-IID partitioning for federated learning.
- For MNIST, images are converted to 3 channels to match the CNN input.
- For CIFAR-10, standard 3-channel images are used.
- The number of clients, local epochs, and other hyperparameters can be set in the config files.
- All configuration files are located in the [`conf/`](conf/) directory.

---

## Code Overview

- [`main.py`](main.py):  
  Orchestrates the experiment using Hydra configs. Handles logging, device selection, and calls the modular factories for data, model, and algorithm setup.

- [`src/dataset_factory.py`](src/dataset_factory.py):  
  Loads datasets (MNIST, CIFAR-10), partitions them into client subsets (IID or Dirichlet non-IID), and returns DataLoaders for federated simulation.

- [`src/model_factory.py`](src/model_factory.py):  
  Defines and instantiates models. Supports a lightweight CNN for MNIST/CIFAR-10 and ResNet18 for CIFAR-10.

- [`src/algorithm_factory.py`](src/algorithm_factory.py):  
  Implements federated learning algorithms (FedAvg, FedProx) using Flower. Handles client/server logic, training, evaluation, and aggregation.

- [`src/utils.py`](src/utils.py):  
  Utility functions for reproducibility, model saving, and evaluation on the MNIST test set.

---

## Device Compatibility

- The code automatically detects and uses the best available device:
  - **Apple Silicon (M1/M2/M3):** Uses MPS acceleration if available.
  - **NVIDIA GPUs:** Uses CUDA if available.
  - **Otherwise:** Falls back to CPU.
- No manual changes are needed; device selection is handled internally in the code.

---

## Troubleshooting

- If you see low accuracy after training, ensure that the model and data channels match and that the global model is properly updated after federated training.
- For more details, see comments in [`main.py`](main.py), [`src/algorithm_factory.py`](src/algorithm_factory.py), and [`src/model_factory.py`](src/model_factory.py).

---

## Authors

...