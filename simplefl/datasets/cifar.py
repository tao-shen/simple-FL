"""
CIFAR Dataset Loader for Simple-FL
"""

import h5py
import numpy as np
import re
from PIL import Image
from simplefl.datasets import dirichlet_split_noniid, init_proxy_data


class CIFAR:
    """
    CIFAR-10/100 dataset loader
    
    Supports natural partitioning, IID, and non-IID (Dirichlet) data splits
    """

    def __init__(self, args, path='./data_in_use/'):
        """
        Initialize CIFAR dataset
        
        Args:
            args: Arguments containing dataset configuration
            path: Path to data directory
        """
        with h5py.File(path + args.dataset + '.h5', 'r') as f:
            train_data, test_data = f['train'][:], f['test'][:]
        
        # Normalize pixel values
        train_data['pixels'] = self.normalize(train_data['pixels'])
        test_data['pixels'] = self.normalize(test_data['pixels'])
        
        # Initialize proxy data if needed
        if args.proxy_ratio > 0:
            train_data, self.proxy_data = init_proxy_data(train_data, args)

        # Partition data based on IID strategy
        if 'natural' in args.iid:
            self.user_offset = {
                'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        
        elif 'alpha' in args.iid:
            try:
                DIRICHLET_ALPHA = float(re.findall(r"\d+\.?\d*", args.iid)[0])
            except:
                DIRICHLET_ALPHA = 0.5
            
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 500
            
            train_labels = train_data['label']
            train_idcs = dirichlet_split_noniid(
                train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            train_data = train_data[np.concatenate(train_idcs)]

            test_labels = test_data['label']
            test_idcs = dirichlet_split_noniid(
                test_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            test_data = test_data[np.concatenate(test_idcs)]
            
            self.user_offset = {
                'train': np.append(0, np.cumsum(list(map(len, train_idcs)))),
                'test': np.append(0, np.cumsum(list(map(len, test_idcs))))}
        
        elif 'iid' in args.iid:
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 3400
            
            self.user_offset = {
                'train': np.linspace(0, len(train_data), N_CLIENTS+1, endpoint=True).astype(int),
                'test': np.linspace(0, len(test_data), N_CLIENTS+1, endpoint=True).astype(int)}

        self.train_data, self.test_data = train_data, test_data

    def normalize(self, img):
        """Normalize image pixels"""
        mean = 120.70748
        std = 64.150024
        # mean = np.mean(img)
        # std = np.std(img)
        img = (img - mean) / std
        return img

    def __getitem__(self, index):
        """Get item by index (for compatibility)"""
        if isinstance(index, int):
            img, target = self.data[index]['pixels'], int(
                self.data[index]['label'])
            img = Image.fromarray(img, mode='F')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            sample = {}
            sample['user_id'], sample['pixels'], sample['label'] = self.data[index]['user_id'], img, target

            return sample
        elif isinstance(index, str):
            return self.data[index]
