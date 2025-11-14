"""
Common utility functions for Simple-FL
"""

import torch
import numpy as np
import random


def setup_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    from torch.backends import cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class Container:
    """
    Container class for wrapping data dictionaries
    """
    def __init__(self, x):
        self.data = x
        self.len = len(list(self.data.values())[0])

    def __getitem__(self, idx):
        x = {}
        for key, value in self.data.items():
            x[key] = value[idx]
        return x

    def __len__(self):
        return self.len


def to_device(x, device):
    """
    Move data to specified device (CPU/GPU)
    
    Args:
        x: Data to move (dict, list, Container, or tensor)
        device: Target device
        
    Returns:
        Data on the specified device
    """
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = value.to(device)
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = to_device(x[i], device)
    elif isinstance(x, Container):
        x = to_device(x.data, device)
    # elif isinstance(x, Container_mcc):
    #     x = to_device(x.data, device)
    else:
        x = x.to(device)
    return x
