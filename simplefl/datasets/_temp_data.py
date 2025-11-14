from locale import normalize
import h5py
import numpy as np
import yaml
import os.path
from PIL import Image
# from torchvision import transforms, utils
from models import *
from utils import Dataset, DataLoader
import re


class Data_init:

    def __init__(self, args, path='./data_in_use/', only_digits=False, **kwargs):
        self.args=args
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
            with h5py.File(args.dataset + '.h5', 'r') as f:
                train_data = f['train'][:]
                test_data = f['test'][:]
                self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        if args.proxy_ratio>0:
            self.proxy_data =data.proxy_data
            
        self.data = data.train_data, data.test_data
        self.user_offset = data.user_offset

class MovieLens:
    def __init__(self, args, path='./data_in_use/'):
        if 'rating' in args.note:
            with h5py.File(path + args.dataset + '.h5', 'r') as f:
                train_data, test_data = f['train'][:], f['test'][:]
                
                if args.proxy_ratio>0:
                    train_data, self.proxy_data=init_proxy_data(train_data, args)
                
                self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
                # self.user_offset = {
                #     'train': f['user_offset']['train'][()], 'test': f['user_offset']['test'][()]}
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
            neg = neg[:, :test_ratio]  # 1:test_ratio
            cand_item = data['cand_item_id'].reshape(-1, 1)
            cand_item_id = np.hstack((cand_item, neg)).flatten()
            label = np.pad(np.ones_like(data['label']).reshape(-1, 1), ((
                0, 0), (0, test_ratio)), constant_values=(0, 0)).flatten()  # 1:test_ratio
            test_data = np.repeat(data, test_ratio+1,
                                  axis=0)  # 1:test_ratio
            test_data['label'] = label
            test_data['cand_item_id'] = cand_item_id
            
            if args.proxy_ratio>0:
                train_data, self.proxy_data=init_proxy_data(train_data, args)
            
            self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
                # self.user_offset = {'train': np.cumsum(np.diff(f['user_offset']['train'], prepend=0)*(train_ratio+1)), 'test': np.cumsum(np.diff(f['user_offset']['test'], prepend=0)*(test_ratio+1))} # 1:4
        self.train_data, self.test_data = train_data, test_data


class Amazon:
    def __init__(self, args, path='./data_in_use/'):
        if 'rating' in args.note:
            with h5py.File(path + args.dataset + '.h5', 'r') as f:
                train_data, test_data = f['train'][:], f['test'][:]
                
                if args.proxy_ratio>0:
                    train_data, self.proxy_data=init_proxy_data(train_data, args)
                
                self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
                # self.user_offset = {
                #     'train': f['user_offset']['train'][()], 'test': f['user_offset']['test'][()]}
        elif 'interaction' in args.note:
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
                neg = neg[:, :test_ratio]  # 1:test_ratio
                cand_item = data['cand_item_id'].reshape(-1, 1)
                cand_item_id = np.hstack((cand_item, neg)).flatten()
                label = np.pad(np.ones_like(data['label']).reshape(-1, 1), ((
                    0, 0), (0, test_ratio)), constant_values=(0, 0)).flatten()  # 1:test_ratio
                test_data = np.repeat(data, test_ratio+1,
                                      axis=0)  # 1:test_ratio
                test_data['label'] = label
                test_data['cand_item_id'] = cand_item_id
                
                if args.proxy_ratio>0:
                    train_data, self.proxy_data=init_proxy_data(train_data, args)
                
                self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                    'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
                # self.user_offset = {'train': np.cumsum(np.diff(f['user_offset']['train'], prepend=0)*(train_ratio+1)), 'test': np.cumsum(np.diff(f['user_offset']['test'], prepend=0)*(test_ratio+1))} # 1:4
        self.train_data, self.test_data = train_data, test_data


class FEMNIST:

    def __init__(self, args, path='./data_in_use/', only_digits=False):

        with h5py.File(path + args.dataset + '.h5', 'r') as f:
            train_data, test_data = f['train'][:], f['test'][:]
        if only_digits:
            train_data = train_data[train_data['label'] < 10]
            test_data = test_data[test_data['label'] < 10]
        train_data['pixels'] = self.normalize(train_data['pixels'])
        test_data['pixels'] = self.normalize(test_data['pixels'])
        
        if args.proxy_ratio>0:
            train_data, self.proxy_data=init_proxy_data(train_data, args)
        
        if 'natural' in args.iid:
            self.user_offset = {
                'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        elif 'alpha' in args.iid:
            try:
                DIRICHLET_ALPHA = float(re.findall(r"\d+\.?\d*", args.iid)[0])
            except:
                pass
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = len(np.unique(train_data['user_id']))
            # DIRICHLET_ALPHA = 1
            train_labels = train_data['label']
            train_idcs = dirichlet_split_noniid(
                train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            train_data = train_data[np.concatenate(train_idcs)]

            test_labels = test_data['label']
            test_idcs = dirichlet_split_noniid(
                test_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            test_data = test_data[np.concatenate(test_idcs)]
            self.user_offset = {'train': np.append(0, np.cumsum(list(map(len, train_idcs)))),
                                'test': np.append(0, np.cumsum(list(map(len, test_idcs))))}
        elif 'iid' in args.iid:
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = len(np.unique(train_data['user_id']))
            self.user_offset = {'train': np.linspace(0, len(train_data), N_CLIENTS+1, endpoint=True).astype(int),
                                'test': np.linspace(0, len(test_data), N_CLIENTS+1, endpoint=True).astype(int)}

        self.train_data, self.test_data = train_data, test_data

    def normalize(self, img):
        mean = 0.03650044
        std = 0.1602474
        # mean = np.mean(img)
        # std = np.std(img)
        img = (img - mean)/std
        return img

    def __getitem__(self, index):
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


class Fashion:

    def __init__(self, args, path='./data_in_use/'):

        with h5py.File(path + args.dataset + '.h5', 'r') as f:
            train_data, test_data = f['train'][:], f['test'][:]
        train_data['pixels'] = self.normalize(train_data['pixels'])
        test_data['pixels'] = self.normalize(test_data['pixels'])
        
        if args.proxy_ratio>0:
            train_data, self.proxy_data=init_proxy_data(train_data, args)
        
        if 'natural' in args.iid:
            self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        elif 'alpha' in args.iid:
            try:
                DIRICHLET_ALPHA = float(re.findall(r"\d+\.?\d*", args.iid)[0])
            except:
                pass
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 3400
            # DIRICHLET_ALPHA = 1
            train_labels = train_data['label']
            train_idcs = dirichlet_split_noniid(
                train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            train_data = train_data[np.concatenate(train_idcs)]

            test_labels = test_data['label']
            test_idcs = dirichlet_split_noniid(
                test_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            test_data = test_data[np.concatenate(test_idcs)]
            self.user_offset = {'train': np.append(0, np.cumsum(list(map(len, train_idcs)))),
                                'test': np.append(0, np.cumsum(list(map(len, test_idcs))))}
        elif 'iid' in args.iid:
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 3400
            self.user_offset = {'train': np.linspace(0, len(train_data), N_CLIENTS+1, endpoint=True).astype(int),
                                'test': np.linspace(0, len(test_data), N_CLIENTS+1, endpoint=True).astype(int)}

        self.train_data, self.test_data = train_data, test_data

    def normalize(self, img):
        mean = 73
        var = 8100
        img = (img - mean)/np.sqrt(var)
        return img

    def __getitem__(self, index):
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


class CIFAR:

    def __init__(self, args, path='./data_in_use/'):

        with h5py.File(path + args.dataset + '.h5', 'r') as f:
            train_data, test_data = f['train'][:], f['test'][:]
        train_data['pixels'] = self.normalize(train_data['pixels'])
        test_data['pixels'] = self.normalize(test_data['pixels'])
        
        if args.proxy_ratio>0:
            train_data, self.proxy_data=init_proxy_data(train_data, args)

        if 'natural' in args.iid:
            self.user_offset = {'train': np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data)),
                                'test': np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))}
        elif 'alpha' in args.iid:
            try:
                DIRICHLET_ALPHA = float(re.findall(r"\d+\.?\d*", args.iid)[0])
            except:
                pass
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 500
            # DIRICHLET_ALPHA = 1
            train_labels = train_data['label']
            train_idcs = dirichlet_split_noniid(
                train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            train_data = train_data[np.concatenate(train_idcs)]

            test_labels = test_data['label']
            test_idcs = dirichlet_split_noniid(
                test_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)
            test_data = test_data[np.concatenate(test_idcs)]
            self.user_offset = {'train': np.append(0, np.cumsum(list(map(len, train_idcs)))),
                                'test': np.append(0, np.cumsum(list(map(len, test_idcs))))}
        elif 'iid' in args.iid:
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            if args.n_clients is not None:
                N_CLIENTS = args.n_clients
            else:
                N_CLIENTS = 3400
            self.user_offset = {'train': np.linspace(0, len(train_data), N_CLIENTS+1, endpoint=True).astype(int),
                                'test': np.linspace(0, len(test_data), N_CLIENTS+1, endpoint=True).astype(int)}

        self.train_data, self.test_data = train_data, test_data

    def normalize(self, img):
        mean = 120.70748
        std = 64.150024
        # mean = np.mean(img)
        # std = np.std(img)
        img = (img - mean)/std
        return img

    def __getitem__(self, index):
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


class Shakespeare:
    
    def __init__(self, args, path='./data_in_use/'):
        self.args = args
        self.path = path
        self.train_data, self.test_data = self.load_data()
        self.user_offset = self.calculate_user_offset(self.train_data, self.test_data)

    def load_data(self):
        """从HDF5文件中加载训练数据和测试数据"""
        with h5py.File(self.path + self.args.dataset + '.h5', 'r') as f:
            train_data = f['train'][:]
            test_data = f['test'][:]
        return train_data, test_data

    def calculate_user_offset(self, train_data, test_data):
        """计算每个用户在数据集中的偏移量，便于之后的数据抽取"""
        user_offset_train = np.append(np.unique(train_data['user_id'], return_index=True)[1], len(train_data))
        user_offset_test = np.append(np.unique(test_data['user_id'], return_index=True)[1], len(test_data))
        return {'train': user_offset_train, 'test': user_offset_test}

    def __getitem__(self, index):
        """使数据集支持索引操作，方便训练时调用"""
        if isinstance(index, int):
            snippet, user_id = self.train_data[index]['snippets'], self.train_data[index]['user_id']
            return {'user_id': user_id, 'snippet': snippet}
        elif isinstance(index, str):
            return self.train_data if index == 'train' else self.test_data


def load_model(args):
    if 'ml-1m' in args.dataset:
        model = DIN(args)
    elif 'femnist' in args.dataset:
        model = LeNet5(args, num_classes=62)
    return model


def iid_each_round(clients, candidates, args):
    loader = [None]*len(clients)

    if 'each_round' in args.iid:
        data = np.concatenate([clients[i].train_data for i in candidates])
        np.random.shuffle(data)
        num_clients = len(candidates)
        idx = np.linspace(0, len(data), num_clients +
                          1, endpoint=True).astype(int)
        for i, id in zip(candidates, range(len(candidates))):
            train_data = data[idx[id]:idx[id+1]]
            loader[i] = DataLoader(
                Dataset(train_data, args), batch_size=args.local_batch_size, shuffle=True)

    return loader


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1
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
    ind = np.random.choice(len(
        train_data), int(args.proxy_ratio*len(train_data)), replace=False)
    proxy_data=train_data[ind]
    train_data=np.delete(train_data, ind)
    for name in proxy_data.dtype.names:
        if 'user' in name:
            proxy_data[name] = 0
    return train_data, proxy_data