# Simple-FL: A Simple and Professional Federated Learning Framework

## Overview

Simple-FL is a clean, modular, and easy-to-use federated learning framework designed for research purposes. It provides implementations of various federated learning algorithms with a focus on code readability and extensibility.

## Features

- 🎯 **Simple & Clean**: Minimal design with clear code structure
- 🔧 **Modular**: Easy to extend with new algorithms, models, and datasets
- 📊 **Multiple Algorithms**: FedAvg, FedProx, Scaffold, FedLeo, and more
- 🗂️ **Multiple Datasets**: FEMNIST, CIFAR-10/100, MovieLens, Fashion-MNIST, etc.
- 🧠 **Flexible Models**: CNN, ResNet, DIN, and custom model support
- 📈 **Experiment Tracking**: Built-in result logging and optional Wandb integration

## Installation

```bash
# Clone the repository
git clone https://github.com/tao-shen/simple-FL.git
cd simple-FL

# Install dependencies
pip install -e .

# Or install with optional dependencies
pip install -e ".[tracking,visualization,notebooks]"
```

## Quick Start

```bash
# Run federated learning experiment
python scripts/train_fl.py

# Run centralized training (baseline)
python scripts/train_centralized.py
```

## Project Structure

```
simple-fl/
├── simplefl/          # Core package
│   ├── methods/       # FL algorithms
│   ├── models/        # Neural network models
│   ├── datasets/      # Dataset loaders
│   ├── core/          # Server, Client, Data initialization
│   └── utils/         # Utility functions
├── scripts/           # Training scripts
├── configs/           # Configuration files
├── data/              # Data directory
├── results/           # Experiment results
└── logs/              # Log files
```

## Configuration

Edit `configs/config.yaml` to customize your experiment:

```yaml
dataset: femnist
device: cuda:0
method: fedavg
local_epochs: 5
clients_per_round: 10
communication_rounds: 100
```

## Supported Algorithms

- **FedAvg**: Federated Averaging
- **FedProx**: Federated Optimization with Proximal Term
- **Scaffold**: Stochastic Controlled Averaging
- **FedLeo**: Federated Learning with Learned Optimizer
- **FedOpt**: Federated Optimization (Adam, Yogi, Adagrad)
- **FedAvgM**: FedAvg with Server Momentum
- And more...

## Supported Datasets

- **FEMNIST**: Federated EMNIST (62 classes)
- **CIFAR-10/100**: Image classification
- **Fashion-MNIST**: Fashion item classification
- **MovieLens**: Recommendation system
- **Amazon**: Product recommendation

## Adding New Algorithms

1. Create a new file in `simplefl/methods/`
2. Inherit from the `FL` base class
3. Implement required methods
4. The algorithm will be automatically discovered

Example:

```python
from simplefl.methods.fl import FL

class MyNewAlgorithm(FL):
    def __init__(self, server, clients, args):
        super().__init__()
        # Your initialization
    
    def train_round(self, round_idx):
        # Your training logic
        pass
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
