"""
Datasets module for Simple-FL

This module contains dataset loaders and data partitioning utilities.
"""

# Import utility functions first (before dataset classes to avoid circular imports)
from .common import dirichlet_split_noniid, init_proxy_data

# Import dataset classes
from .femnist import FEMNIST
from .cifar import CIFAR
from .movielens import MovieLens
from .amazon import Amazon
from .fashion import Fashion
from .shakespeare import Shakespeare

__all__ = [
    'FEMNIST',
    'CIFAR',
    'MovieLens',
    'Amazon',
    'Fashion',
    'Shakespeare',
    'dirichlet_split_noniid',
    'init_proxy_data',
]
