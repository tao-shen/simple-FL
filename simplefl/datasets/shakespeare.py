"""
Shakespeare Dataset Loader for Simple-FL
"""

import h5py
import numpy as np


class Shakespeare:
    """
    Shakespeare dataset loader for text-based federated learning
    """
    
    def __init__(self, args, path='./data_in_use/'):
        """
        Initialize Shakespeare dataset
        
        Args:
            args: Arguments containing dataset configuration
            path: Path to data directory
        """
        self.args = args
        self.path = path
        self.train_data, self.test_data = self.load_data()
        self.user_offset = self.calculate_user_offset(self.train_data, self.test_data)

    def load_data(self):
        """
        Load training and test data from HDF5 file
        
        Returns:
            Tuple of (train_data, test_data)
        """
        with h5py.File(self.path + self.args.dataset + '.h5', 'r') as f:
            train_data = f['train'][:]
            test_data = f['test'][:]
        return train_data, test_data

    def calculate_user_offset(self, train_data, test_data):
        """
        Calculate user offsets for data partitioning
        
        Args:
            train_data: Training data array
            test_data: Test data array
            
        Returns:
            Dictionary with 'train' and 'test' user offsets
        """
        user_offset_train = np.append(
            np.unique(train_data['user_id'], return_index=True)[1], 
            len(train_data))
        user_offset_test = np.append(
            np.unique(test_data['user_id'], return_index=True)[1], 
            len(test_data))
        return {'train': user_offset_train, 'test': user_offset_test}

    def __getitem__(self, index):
        """
        Get item by index (for compatibility)
        """
        if isinstance(index, int):
            snippet, user_id = self.train_data[index]['snippets'], self.train_data[index]['user_id']
            return {'user_id': user_id, 'snippet': snippet}
        elif isinstance(index, str):
            return self.train_data if index == 'train' else self.test_data

