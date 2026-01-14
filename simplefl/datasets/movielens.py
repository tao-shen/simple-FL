"""
MovieLens Dataset Loader for Simple-FL
"""

import h5py
import numpy as np
from .common import init_proxy_data


class MovieLens:
    """MovieLens dataset loader for recommendation tasks"""

    def __init__(self, args, path='./data/'):
        if 'rating' in args.note:
            with h5py.File(path + args.dataset + '.h5', 'r') as f:
                train_data, test_data = f['train'][:], f['test'][:]
                
                if args.proxy_ratio > 0:
                    train_data, self.proxy_data = init_proxy_data(train_data, args)
                
                self.user_offset = {
                    'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        # elif 'interaction' in args.note:
        train_ratio, test_ratio = 4, 99
        with h5py.File(path + args.dataset + '.h5', 'r') as f:
            # train_data 1:train_ratio
            data = f['train'][:]
            neg = np.hstack(f['user_neg_item']['train']).reshape(-1, 4)
            neg = neg[:, :train_ratio]
            cand_item = data['cand_item_id'].reshape(-1, 1)
            cand_item_id = np.hstack((cand_item, neg)).flatten()
            label = np.pad(np.ones_like(
                data['label']).reshape(-1, 1), ((0, 0), (0, train_ratio)), constant_values=(0, 0)).flatten()
            train_data = np.repeat(data, train_ratio+1, axis=0)
            train_data['label'] = label
            train_data['cand_item_id'] = cand_item_id
            # test_data 1:test_ratio
            data = f['test'][:]
            neg = np.hstack(f['user_neg_item']['test']).reshape(-1, 99)
            neg = neg[:, :test_ratio]
            cand_item = data['cand_item_id'].reshape(-1, 1)
            cand_item_id = np.hstack((cand_item, neg)).flatten()
            label = np.pad(np.ones_like(data['label']).reshape(-1, 1), ((
                0, 0), (0, test_ratio)), constant_values=(0, 0)).flatten()
            test_data = np.repeat(data, test_ratio+1, axis=0)
            test_data['label'] = label
            test_data['cand_item_id'] = cand_item_id
            
            if args.proxy_ratio > 0:
                train_data, self.proxy_data = init_proxy_data(train_data, args)
            
            self.user_offset = {
                'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        
        self.train_data, self.test_data = train_data, test_data
