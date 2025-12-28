"""
Simple-FL: A Simple and Professional Federated Learning Framework

This package provides a clean, modular implementation of federated learning
algorithms for research purposes.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core components
from .core import Server, Client, init_clients, Data_init
from .methods import fl_methods
from .utils import init_args, setup_seed, save_results

__all__ = [
    'Server',
    'Client', 
    'init_clients',
    'Data_init',
    'fl_methods',
    'init_args',
    'setup_seed',
    'save_results',
]
