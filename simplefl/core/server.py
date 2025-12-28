"""
Server module for Simple-FL
"""

import torch
from torch.utils.data import DataLoader
from simplefl.utils.config import Dataset
from simplefl.models import *


class Server:
    """
    Federated Learning Server
    
    The server maintains the global model and coordinates training across clients.
    """

    def __init__(self, init, args):
        """
        Initialize server
        
        Args:
            init: Data initialization object containing train/test data
            args: Arguments containing server configuration
        """
        # self.features = args.features
        self.args = args
        self.init = init
        self.train_data, self.test_data = init.data
        train_set = Dataset(self.train_data, args)
        test_set = Dataset(self.test_data, args)
        self.train_loader = DataLoader(
            train_set, batch_size=args.server_batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_set, batch_size=args.eval_batch_size)
    
    def init_proxy_data(self):
        """
        Initialize proxy data for server-side training (e.g., for FedLeo)
        
        Returns:
            Proxy data array
        """
        print('loading proxy data...')
        labels = self.init.proxy_data['label']

        # # 确定类别数量
        # num_classes = len(np.unique(labels))

        # # 创建一个字典来存储每个类别的选择样本
        # ind = []

        # # 对于每个类别，选择一个样本
        # for class_label in range(num_classes):
        #     # 找到属于当前类别的样本索引
        #     class_indices = np.where(labels == class_label)[0]
            
        #     # 从当前类别的样本中选择一个样本
        #     selected_sample_index = ind.append(np.random.choice(class_indices))
        
        # self.init.proxy_data=self.init.proxy_data[ind]
        return self.init.proxy_data
        

    def init_model(self):
        """
        Initialize model based on dataset type
        
        Returns:
            Initialized model
        """
        if 'ml-1m' in self.args.dataset:
            model = DIN(self.args)
        elif self.args.dataset == 'cifar10':
            model = ResNet20(self.args, num_classes=10)
        elif self.args.dataset == 'cifar100':
            model = ResNet20(self.args, num_classes=100)
        elif 'femnist' in self.args.dataset:
            model = CNN_FEMNIST(self.args, num_classes=62)
        elif 'fashionmnist' in self.args.dataset:
            model = LeNet5(self.args)
        return model
