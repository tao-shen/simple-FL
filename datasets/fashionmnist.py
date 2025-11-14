from torchvision.datasets import MNIST, EMNIST, utils, FashionMNIST
import os.path
import h5py
import numpy as np


def data_process():
    path = './data_in_use/'
    train = FashionMNIST(path, train=True, download=True)
    test = FashionMNIST(path, train=False, download=True)
    with h5py.File(path+'fashionmnist.h5', 'w') as f:
        pixels = list(train.data)
        labels = list(train.targets)
        train = list(map(tuple, zip(pixels, labels)))

        pixels = list(test.data)
        labels = list(test.targets)
        test = list(map(tuple, zip(pixels, labels)))
        dt = {'names': ['pixels', 'label'],
              'formats': ['(1,28,28)float32', 'int64']}
        trainset = np.array(train, dtype=dt)
        testset = np.array(test, dtype=dt)
        trainset = f.create_dataset('train', data=trainset)
        testset = f.create_dataset('test', data=testset)


data_process()
