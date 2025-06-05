# OptML-Project

Federated learning experiments on MNIST and CIFAR-10 using PyTorch and Flower.

## Project Structure

```
.
├── algorithm_factory.py
├── dataset_factory.py
├── main.py
├── model_factory.py
├── TrainedModelEval.ipynb
├── conf/
│   ├── mnist_cnn.yaml
│   ├── mnist_cnn_iid.yaml
│   ├── cifar_resnet18.yaml
│   ├── cifar_resnet18_iid.yaml
│   └── ...
├── TrainedModels/
│   └── ...
├── data/
│   ├── MNIST/
│   └── cifar-10-batches-py/
└── outputs/
```

## Setup

1. **Install dependencies**  
   Recommended: use a virtual environment (e.g., conda or venv).

   ```sh
   pip install torch torchvision hydra-core flwr fedlab matplotlib numpy
   ```

   #### Library Versions

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

## Training

All training is launched via `main.py` using Hydra configs from the `conf` folder.

### Example: Train CNN on MNIST (IID)

```sh
python main.py --config-name=mnist_cnn_iid.yaml
```

### Example: Train CNN on MNIST (Dirichlet non-IID)

```sh
python main.py --config-name=mnist_cnn.yaml
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

## Evaluation

Use the provided Jupyter notebook [`TrainedModelEval.ipynb`](TrainedModelEval.ipynb) to evaluate and visualize the trained models.

1. Open the notebook in VS Code or Jupyter.
2. Adjust the `model_path` in the relevant cells to point to your trained model file in `TrainedModels/`.
3. Run the notebook cells to compute accuracy and visualize predictions.

## Configuration

All experiment settings (dataset, model, algorithm, etc.) are controlled via YAML files in the `conf/` directory.  
You can create new configs by copying and modifying existing ones.

## Notes

- The code supports both IID and Dirichlet non-IID partitioning for federated learning.
- For MNIST, images are converted to 3 channels to match the CNN input.
- For CIFAR-10, standard 3-channel images are used.
- The number of clients, local epochs, and other hyperparameters can be set in the config files.

## Troubleshooting

- If you see low accuracy after training, ensure that the model and data channels match and that the global model is properly updated after federated training.
- For more details, see comments in [`main.py`](main.py), [`algorithm_factory.py`](algorithm_factory.py), and [`model_factory.py`](model_factory.py).

---

**Author:**  
