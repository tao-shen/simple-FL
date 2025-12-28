"""
Data Initialization Module for Simple-FL

This module handles dataset loading and model initialization based on dataset type.
"""

import h5py
import numpy as np

# Import dataset classes
from simplefl.datasets import (
    FEMNIST, CIFAR, MovieLens, Amazon, Fashion, Shakespeare
)

# Import model classes
from simplefl.models.din import DIN
from simplefl.models.cnn import CNN_FEMNIST, ResNet18
from simplefl.models.resnet import ResNet20
from simplefl.models.mlp import MLP_Mixer


class Data_init:
    """
    Data initialization class that loads datasets and initializes models
    """

    def __init__(self, args, path='./data_in_use/', only_digits=False, **kwargs):
        """
        Initialize data and model based on dataset type
        
        Args:
            args: Arguments containing dataset and model configuration
            path: Path to data directory
            only_digits: For FEMNIST, whether to use only digit classes
        """
        self.args = args
        
        # Initialize dataset and model based on dataset type
        if 'ml-1m' in args.dataset:
            data = MovieLens(args)
            args.recorder = {'loss': [], 'hit5': [], 'recall5': [], 'ndcg5': [],
                             'auc': [], 'hit10': [], 'recall10': [], 'ndcg10': []}
            self.model = DIN(args)
        
        elif 'ml-100k' in args.dataset:
            data = MovieLens(args)
            args.recorder = {'loss': [], 'hit5': [], 'recall5': [], 'ndcg5': [],
                             'auc': [], 'hit10': [], 'recall10': [], 'ndcg10': []}
            self.model = DIN(args)
        
        elif 'amazon' in args.dataset:
            data = Amazon(args)
            args.recorder = {'loss': [], 'hit5': [], 'recall5': [], 'ndcg5': [],
                             'auc': [], 'hit10': [], 'recall10': [], 'ndcg10': []}
            self.model = DIN(args)
        
        elif 'femnist' in args.dataset:
            data = FEMNIST(args, only_digits=only_digits)
            args.recorder = {'loss': [], 'acc': []}
            if only_digits:
                # self.model = LeNet5(args,num_classes=10)
                self.model = CNN_FEMNIST(args, num_classes=10)
            else:
                # self.model = LeNet5(args, num_classes=62)
                self.model = CNN_FEMNIST(args, num_classes=62)

        elif 'fashionmnist' in args.dataset:
            data = Fashion(args)
            args.recorder = {'loss': [], 'acc': []}
            # self.model = LeNet5(args, num_classes=10)
            self.model = MLP_Mixer(args, num_classes=10)

        elif 'cifar10' in args.dataset:
            data = CIFAR(args)
            args.recorder = {'loss': [], 'acc': []}
            self.model = ResNet20(args, num_classes=10)

        elif 'cifar100' in args.dataset:
            data = CIFAR(args)
            args.recorder = {'loss': [], 'acc': []}
            self.model = ResNet20(args, num_classes=100)
            
        elif 'shakespeare' in args.dataset:
            data = Shakespeare(args)
            args.recorder = {'loss': [], 'acc': []}
            self.model = ResNet18(args, num_classes=100)    

        else:
            # Generic dataset loading
            with h5py.File(args.dataset + '.h5', 'r') as f:
                train_data = f['train'][:]
                test_data = f['test'][:]
                self.user_offset = {
                    'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        
        # Store proxy data if available
        if args.proxy_ratio > 0:
            self.proxy_data = data.proxy_data
            
        # Store train/test data and user offsets
        self.data = data.train_data, data.test_data
        self.user_offset = data.user_offset
