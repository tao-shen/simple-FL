# Simple-FL: A Simple and Professional Federated Learning Framework

[English](#english) | [中文](#中文)

---

## English

### Overview

Simple-FL is a clean, modular, and easy-to-use federated learning framework designed for research purposes. It provides implementations of various federated learning algorithms with a focus on code readability and extensibility.

### Features

- 🎯 **Simple & Clean**: Minimal design with clear code structure
- 🔧 **Modular**: Easy to extend with new algorithms, models, and datasets
- 📊 **Multiple Algorithms**: FedAvg, FedProx, Scaffold, FedLeo, and more
- 🗂️ **Multiple Datasets**: FEMNIST, CIFAR-10/100, MovieLens, Fashion-MNIST, etc.
- 🧠 **Flexible Models**: CNN, ResNet, DIN, and custom model support
- 📈 **Experiment Tracking**: Built-in result logging and optional Wandb integration

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd simple-fl

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run federated learning experiment
python scripts/train_fl.py

# Run centralized training (baseline)
python scripts/train_centralized.py
```

### Project Structure

```
simple-fl/
├── simplefl/          # Core package
│   ├── methods/       # FL algorithms
│   ├── models/        # Neural network models
│   ├── datasets/      # Dataset loaders
│   ├── core/          # Server, Client, Data initialization
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── scripts/           # Training scripts
├── data/              # Data directory
├── results/           # Experiment results
└── logs/              # Log files
```


### Configuration

Edit `configs/config.yaml` to customize your experiment:

```yaml
dataset: femnist
device: cuda:0
method: fedavg
local_epochs: 5
clients_per_round: 10
communication_rounds: 100
```

### Supported Algorithms

- **FedAvg**: Federated Averaging
- **FedProx**: Federated Optimization with Proximal Term
- **Scaffold**: Stochastic Controlled Averaging
- **FedLeo**: Federated Learning with Learned Optimizer
- And more...

### Supported Datasets

- **FEMNIST**: Federated EMNIST (62 classes)
- **CIFAR-10/100**: Image classification
- **Fashion-MNIST**: Fashion item classification
- **MovieLens**: Recommendation system
- **Amazon**: Product recommendation

### Citation

If you use this code in your research, please cite:

```bibtex
@misc{simple-fl,
  title={Simple-FL: A Simple and Professional Federated Learning Framework},
  author={Your Name},
  year={2024}
}
```

---

## 中文

### 概述

Simple-FL 是一个简洁、模块化、易于使用的联邦学习框架，专为科研目的设计。它提供了多种联邦学习算法的实现，注重代码可读性和可扩展性。

### 特性

- 🎯 **简单清晰**：最小化设计，代码结构清晰
- 🔧 **模块化**：易于扩展新算法、模型和数据集
- 📊 **多种算法**：FedAvg、FedProx、Scaffold、FedLeo 等
- 🗂️ **多种数据集**：FEMNIST、CIFAR-10/100、MovieLens、Fashion-MNIST 等
- 🧠 **灵活模型**：CNN、ResNet、DIN 及自定义模型支持
- 📈 **实验追踪**：内置结果记录和可选的 Wandb 集成

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd simple-fl

# 安装依赖
pip install -r requirements.txt
```

### 快速开始

```bash
# 运行联邦学习实验
python scripts/train_fl.py

# 运行中心化训练（基线）
python scripts/train_centralized.py
```

### 配置

编辑 `configs/config.yaml` 自定义实验：

```yaml
dataset: femnist
device: cuda:0
method: fedavg
local_epochs: 5
clients_per_round: 10
communication_rounds: 100
```

### 许可证

MIT License

### 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
