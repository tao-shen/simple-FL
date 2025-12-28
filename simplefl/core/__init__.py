"""
Core module for Simple-FL

This module contains the core components of the federated learning framework.
"""

from .server import Server
from .client import Client, init_clients
from .data_init import Data_init

__all__ = ['Server', 'Client', 'init_clients', 'Data_init']
