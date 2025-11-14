# Changelog

All notable changes to Simple-FL will be documented in this file.

## [0.1.0] - 2024-11-14

### Added
- Initial release of Simple-FL framework
- Restructured codebase with professional modular architecture
- Created `simplefl` package with clear module separation:
  - `simplefl.core`: Server, Client, and Data initialization
  - `simplefl.methods`: FL algorithms (FedAvg, FedProx, Scaffold, FedLeo, etc.)
  - `simplefl.models`: Neural network models (CNN, ResNet, DIN, etc.)
  - `simplefl.datasets`: Dataset loaders (FEMNIST, CIFAR, MovieLens, etc.)
  - `simplefl.utils`: Utility functions (config, results, common tools)
- New training scripts in `scripts/` directory
- Comprehensive README with installation and usage instructions
- Project configuration files (requirements.txt, .gitignore)

### Changed
- Reorganized all code from flat structure to modular package structure
- Moved training scripts to `scripts/` directory
- Updated all import paths to use `simplefl.*` namespace
- Preserved all original functionality and algorithms
- Kept all commented code as备选方案 (alternative options)

### Migration Notes
- Old `fl_training.py` → `scripts/train_fl.py`
- Old `centralized_training.py` → `scripts/train_centralized.py`
- Old `data.py` → `simplefl/datasets/*.py` + `simplefl/core/data_init.py`
- Old `server_client.py` → `simplefl/core/server.py` + `simplefl/core/client.py`
- Old `utils.py` → `simplefl/utils/*.py`
- Old `methods/*.py` → `simplefl/methods/*.py`
- Old `models/*.py` → `simplefl/models/*.py`

### Technical Details
- Python 3.7+ required
- PyTorch 1.9+ required
- All original features preserved
- Dynamic algorithm loading maintained
- Configuration system unchanged
- Database logging functionality preserved

## Future Plans
- Add comprehensive unit tests
- Add API documentation
- Add tutorial notebooks
- Performance optimizations
- Additional FL algorithms
