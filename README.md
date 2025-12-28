# Simple-FL: A Simple and Professional Federated Learning Framework

## Overview

Simple-FL is a clean, modular, and easy-to-use federated learning framework designed for research purposes. It provides implementations of various federated learning algorithms with a focus on code readability and extensibility.

## Features

- 🎯 **Simple & Clean**: Minimal design with clear code structure and comprehensive documentation
- 🔧 **Modular Architecture**: Easy to extend with new algorithms, models, and datasets
- 📊 **Comprehensive Algorithms**: 20+ federated learning algorithms implemented
- 🗂️ **Multiple Datasets**: Support for 15+ datasets across various domains
- 🧠 **Flexible Models**: CNN, ResNet, DIN, and custom model support
- 📈 **Experiment Tracking**: Built-in result logging and optional Wandb integration
- 🔬 **Research-Ready**: Designed for reproducible research and fair comparisons

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/tao-shen/simple-FL.git
cd simple-FL

# Install dependencies
pip install -e .
```

### Installation with Optional Dependencies

```bash
# Install with experiment tracking (Wandb)
pip install -e ".[tracking]"

# Install with visualization tools
pip install -e ".[visualization]"

# Install with Jupyter notebook support
pip install -e ".[notebooks]"

# Install all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Run Federated Learning Experiment

```bash
# Run federated learning with default configuration
python scripts/train_fl.py

# Run with custom configuration
python scripts/train_fl.py --config configs/config.yaml
```

### Run Centralized Training (Baseline)

```bash
# Run centralized training for comparison
python scripts/train_centralized.py
```

## Project Structure

```
simple-fl/
├── simplefl/              # Core package
│   ├── methods/           # FL algorithm implementations
│   ├── models/            # Neural network models
│   ├── datasets/          # Dataset loaders and utilities
│   ├── core/              # Server, Client, Data initialization
│   └── utils/             # Utility functions (config, logging, etc.)
├── scripts/               # Training and evaluation scripts
├── configs/               # Configuration files
│   ├── config.yaml        # Main experiment configuration
│   ├── methods.yaml       # Algorithm-specific hyperparameters
│   └── datasets.yaml      # Dataset-specific configurations
├── data/                  # Data directory (created automatically)
├── results/               # Experiment results and logs
└── logs/                  # Log files
```

## Configuration

Edit `configs/config.yaml` to customize your experiment:

```yaml
dataset: femnist              # Dataset name
device: cuda:0                # Device (cuda:0, cpu, etc.)
method: fedavg                # FL algorithm
local_epochs: 5               # Local training epochs
clients_per_round: 10         # Number of clients per round
communication_rounds: 100     # Total communication rounds
lr_l: 0.01                    # Local learning rate
```

Algorithm-specific hyperparameters are configured in `configs/methods.yaml`, and dataset-specific settings are in `configs/datasets.yaml`.

## Supported Algorithms

Simple-FL implements a comprehensive collection of federated learning algorithms:

### Core Algorithms

- **FedAvg** [[McMahan et al., 2017]](https://arxiv.org/abs/1602.05629): Federated Averaging - The foundational algorithm for federated learning
- **FedAvgM**: FedAvg with Server Momentum - Improves convergence with momentum on the server
- **FedProx** [[Li et al., 2020]](https://arxiv.org/abs/1812.06127): Federated Optimization with Proximal Term - Handles statistical and systems heterogeneity
- **Scaffold** [[Karimireddy et al., 2020]](https://arxiv.org/abs/1910.06378): Stochastic Controlled Averaging - Uses control variates to correct client drift

### Advanced Optimization Algorithms

- **FedOpt**: Federated Optimization - Supports Adam, Yogi, and Adagrad server optimizers
- **FedDyn** [[Acar et al., 2021]](https://arxiv.org/abs/2111.04263): Federated Dynamic Regularization - Dynamic regularization for better convergence
- **MIME** [[Karimireddy et al., 2020]](https://arxiv.org/abs/2008.03606): Mimicking Centralized SGD - Mimics centralized optimization in federated setting

### Knowledge Distillation & Meta-Learning

- **FedDF** [[Lin et al., 2020]](https://arxiv.org/abs/1911.00643): Federated Distillation - Uses knowledge distillation for model aggregation
- **FedMeta**: Federated Meta-Learning - Applies meta-learning principles to federated learning
- **FedLeo**: Federated Learning with Learned Optimizer - Uses learned optimizers for better adaptation

### Specialized Algorithms

- **FedEve**: Federated Learning with Efficient Updates - Optimized for communication efficiency
- **FedSpeed**: Fast Federated Learning - Accelerates federated training
- **FedRecom**: Federated Recommendation - Specialized for recommendation systems
- **FedCSD**: Federated Learning with Client-Specific Decay - Client-specific learning rate decay
- **FedCDA**: Federated Learning with Client Data Augmentation - Data augmentation at clients
- **FedLow**: Federated Learning with Low-Rank Adaptation - Uses LoRA for parameter-efficient training
- **Elastic**: Elastic Federated Learning - Adaptive client selection and aggregation

### Variants & Extensions

- **FedAvg_Client_Drift_Only**: FedAvg variant focusing on client drift analysis
- **FedAvg_Period_Drift_Only**: FedAvg variant with periodic drift analysis
- **Gaussian**: Gaussian-based aggregation method

### Algorithm Comparison

| Algorithm | Key Feature | Use Case |
|-----------|-------------|----------|
| FedAvg | Standard averaging | Baseline, general purpose |
| FedProx | Proximal term | Non-IID data, heterogeneous systems |
| Scaffold | Control variates | Statistical heterogeneity |
| FedOpt | Adaptive server optimizer | Faster convergence |
| FedDF | Knowledge distillation | Model compression, heterogeneous models |
| FedLeo | Learned optimizer | Fast adaptation, few-shot learning |
| FedLow | LoRA adaptation | Large models, parameter efficiency |

## Supported Datasets

Simple-FL supports a wide range of datasets across different domains:

### Computer Vision

- **FEMNIST**: Federated EMNIST (62 classes) - Handwritten character recognition
- **CIFAR-10**: 10-class image classification
- **CIFAR-100**: 100-class image classification
- **Fashion-MNIST**: Fashion item classification (10 classes)
- **TinyImageNet**: ImageNet subset (200 classes)

### Natural Language Processing

- **Shakespeare**: Next-character prediction on Shakespeare's works
- **Stack Overflow**: Text classification and tagging

### Recommendation Systems

- **MovieLens-1M**: Movie recommendation (1 million ratings)
- **MovieLens-100K**: Movie recommendation (100K ratings)
- **Amazon**: Product recommendation
- **Last.fm**: Music recommendation


### Dataset Characteristics

| Dataset | Type | Classes/Tasks | Typical Use Case |
|---------|------|--------------|------------------|
| FEMNIST | Vision | 62 classes | Character recognition, non-IID |
| CIFAR-10/100 | Vision | 10/100 classes | Image classification |
| Fashion-MNIST | Vision | 10 classes | Fashion classification |
| MovieLens | Recommendation | Rating prediction | Collaborative filtering |
| Shakespeare | NLP | Next-char prediction | Sequential modeling |
| Stack Overflow | NLP | Text classification | Multi-label classification |

## Usage Examples

### Basic Training

```python
# Configure your experiment in configs/config.yaml
# Then run:
python scripts/train_fl.py
```

### Custom Configuration

```python
# Edit configs/config.yaml
dataset: cifar10
method: fedprox
local_epochs: 5
clients_per_round: 10
communication_rounds: 200
```

### Algorithm-Specific Hyperparameters

Edit `configs/methods.yaml` to tune algorithm-specific parameters:

```yaml
cifar10:
  fedprox:
    lr_l: 0.01
    mu: 1.0          # Proximal term coefficient
  fedopt:
    lr_l: 0.01
    lr_g: 0.001      # Server learning rate
    beta1: 0.9
    beta2: 0.99
    tau: 0.0001
```

## Adding New Algorithms

Simple-FL makes it easy to add new algorithms. Follow these steps:

1. **Create a new file** in `simplefl/methods/` (e.g., `my_algorithm.py`)

2. **Inherit from the base class**:
   - Inherit from `FL` for basic algorithms
   - Inherit from `FedAvg` for FedAvg-based algorithms
   - Inherit from `FedAvgM` for momentum-based algorithms

3. **Implement required methods**:
   - `__init__`: Initialize your algorithm
   - `train_round`: Implement one round of training

4. **Register the algorithm**: The algorithm will be automatically discovered if the class name matches the filename (case-insensitive)

Example:

```python
from simplefl.methods.fl import FL

class MyNewAlgorithm(FL):
    def __init__(self, server, clients, args):
        super().__init__()
        self.server = server
        self.clients = clients
        self.args = args
        # Your initialization code
    
    def train_round(self, round_idx):
        # Your training logic for one round
        # 1. Select clients
        # 2. Local training
        # 3. Aggregate updates
        # 4. Update server model
        pass
```



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This framework implements algorithms from various research papers. Please refer to the original papers for theoretical foundations and cite them appropriately when using specific algorithms.

## Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/tao-shen/simple-FL/issues](https://github.com/tao-shen/simple-FL/issues)
- Repository: [https://github.com/tao-shen/simple-FL](https://github.com/tao-shen/simple-FL)

---

**Note**: This framework is designed for research purposes. For production deployments, additional considerations for security, privacy, and scalability should be addressed.
