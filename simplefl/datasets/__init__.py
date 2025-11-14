"""
Datasets module for Simple-FL

This module contains dataset loaders and data partitioning utilities.
"""

import numpy as np


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    Split data using Dirichlet distribution for non-IID partitioning
    
    Args:
        train_labels: Array of training labels
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
        n_clients: Number of clients
        
    Returns:
        List of client data indices
    """
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def init_proxy_data(train_data, args):
    """
    Initialize proxy data for server-side training (e.g., for FedLeo)
    
    Args:
        train_data: Training data array
        args: Arguments containing proxy_ratio
        
    Returns:
        Tuple of (remaining_train_data, proxy_data)
    """
    ind = np.random.choice(len(
        train_data), int(args.proxy_ratio*len(train_data)), replace=False)
    proxy_data = train_data[ind]
    train_data = np.delete(train_data, ind)
    # Set user_id to 0 for proxy data (make it user-agnostic)
    for name in proxy_data.dtype.names:
        if 'user' in name:
            proxy_data[name] = 0
    return train_data, proxy_data
