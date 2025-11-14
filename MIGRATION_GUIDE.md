# Simple-FL 迁移指南 / Migration Guide

## 概述 / Overview

本文档说明如何从旧代码结构迁移到新的 Simple-FL 框架。

This document explains how to migrate from the old code structure to the new Simple-FL framework.

---

## 主要变更 / Major Changes

### 1. 目录结构 / Directory Structure

**旧结构 / Old Structure:**
```
.
├── methods/
├── models/
├── data.py
├── server_client.py
├── utils.py
├── fl_training.py
└── centralized_training.py
```

**新结构 / New Structure:**
```
.
├── simplefl/                    # 主包 / Main package
│   ├── methods/                 # 算法 / Algorithms
│   ├── models/                  # 模型 / Models
│   ├── datasets/                # 数据集 / Datasets
│   ├── core/                    # 核心组件 / Core components
│   └── utils/                   # 工具函数 / Utilities
├── scripts/                     # 运行脚本 / Run scripts
│   ├── train_fl.py
│   └── train_centralized.py
├── configs/                     # 配置文件 / Config files
├── results/                     # 实验结果 / Results
└── logs/                        # 日志 / Logs
```

### 2. 导入路径变更 / Import Path Changes

| 旧导入 / Old Import | 新导入 / New Import |
|---------------------|---------------------|
| `from data import Data_init` | `from simplefl.core import Data_init` |
| `from server_client import Server, Client` | `from simplefl.core import Server, Client` |
| `from methods import fl_methods` | `from simplefl.methods import fl_methods` |
| `from utils import init_args, setup_seed` | `from simplefl.utils import init_args, setup_seed` |
| `from models import CNN_FEMNIST` | `from simplefl.models import CNN_FEMNIST` |

### 3. 运行脚本变更 / Script Changes

**旧方式 / Old Way:**
```bash
python fl_training.py
python centralized_training.py
```

**新方式 / New Way:**
```bash
python scripts/train_fl.py
python scripts/train_centralized.py
```

---

## 迁移步骤 / Migration Steps

### 步骤 1: 更新导入 / Step 1: Update Imports

如果你有自定义脚本，需要更新导入语句：

If you have custom scripts, update the import statements:

```python
# 旧代码 / Old code
from data import Data_init
from server_client import Server, init_clients
from methods import fl_methods
from utils import init_args, setup_seed

# 新代码 / New code
from simplefl.core import Data_init, Server, init_clients
from simplefl.methods import fl_methods
from simplefl.utils import init_args, setup_seed
```

### 步骤 2: 更新配置 / Step 2: Update Configuration

配置文件位置和格式保持不变，无需修改。

Configuration files location and format remain unchanged, no modifications needed.

### 步骤 3: 测试导入 / Step 3: Test Imports

运行测试脚本验证所有导入正常：

Run the test script to verify all imports work:

```bash
python scripts/test_import.py
```

### 步骤 4: 运行实验 / Step 4: Run Experiments

使用新的脚本路径运行实验：

Use the new script paths to run experiments:

```bash
# 联邦学习 / Federated Learning
python scripts/train_fl.py

# 中心化训练 / Centralized Training
python scripts/train_centralized.py
```

---

## 可选依赖 / Optional Dependencies

以下依赖是可选的，代码会自动检测并优雅降级：

The following dependencies are optional, code will auto-detect and gracefully degrade:

- **wandb**: 实验追踪 / Experiment tracking
- **peft**: FedLow 算法需要 / Required for FedLow algorithm
- **pymysql**: 远程数据库日志 / Remote database logging

---

## 兼容性说明 / Compatibility Notes

### 保留的功能 / Preserved Features

✅ 所有算法实现保持不变 / All algorithm implementations unchanged
✅ 所有模型架构保持不变 / All model architectures unchanged
✅ 配置系统保持不变 / Configuration system unchanged
✅ 数据加载逻辑保持不变 / Data loading logic unchanged
✅ 结果保存功能保持不变 / Results saving functionality unchanged
✅ 所有注释代码保留 / All commented code preserved

### 新增功能 / New Features

✨ 模块化包结构 / Modular package structure
✨ 清晰的命名空间 / Clear namespaces
✨ 更好的代码组织 / Better code organization
✨ 完整的文档 / Complete documentation
✨ 导入测试脚本 / Import test script

---

## 常见问题 / FAQ

### Q: 旧代码还能用吗？/ Can I still use the old code?

A: 可以。旧代码文件仍然保留在根目录，但建议使用新的 `simplefl` 包。

Yes. Old code files are still in the root directory, but we recommend using the new `simplefl` package.

### Q: 如何添加新算法？/ How to add a new algorithm?

A: 在 `simplefl/methods/` 目录下创建新文件，继承 `FL` 基类。动态加载机制会自动识别。

Create a new file in `simplefl/methods/`, inherit from `FL` base class. Dynamic loading will auto-detect it.

### Q: 配置文件需要修改吗？/ Do I need to modify config files?

A: 不需要。配置文件格式和位置保持不变。

No. Config file format and location remain unchanged.

### Q: 如何运行特定算法？/ How to run a specific algorithm?

A: 修改 `configs/config.yaml` 中的 `method` 参数，或使用命令行参数：

Modify the `method` parameter in `configs/config.yaml`, or use command line:

```bash
python scripts/train_fl.py --method=fedavg
python scripts/train_fl.py --method=fedprox
python scripts/train_fl.py --method=scaffold
```

---

## 获取帮助 / Getting Help

如有问题，请：
- 查看 README.md
- 运行 `python scripts/test_import.py` 检查环境
- 提交 Issue

For questions:
- Check README.md
- Run `python scripts/test_import.py` to check environment
- Submit an Issue
