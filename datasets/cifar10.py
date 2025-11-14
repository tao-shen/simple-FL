from torchvision.datasets import MNIST, EMNIST, utils, FashionMNIST, CIFAR10
import os.path
import h5py
import numpy as np


def data_process():
    path = './data_in_use/'
    train = CIFAR10(path, train=True, download=True)
    train.data = np.swapaxes(train.data, 2, 3)
    train.data = np.swapaxes(train.data, 1, 2)
    test = CIFAR10(path, train=False, download=True)
    test.data = np.swapaxes(test.data, 2, 3)
    test.data = np.swapaxes(test.data, 1, 2)
    with h5py.File(path+'cifar10.h5', 'w') as f:
        pixels = list(train.data)
        labels = list(train.targets)
        train = list(map(tuple, zip(pixels, labels)))

        pixels = list(test.data)
        labels = list(test.targets)
        test = list(map(tuple, zip(pixels, labels)))
        dt = {'names': ['pixels', 'label'],
              'formats': ['(3,32,32)float32', 'int64']}
        trainset = np.array(train, dtype=dt)
        testset = np.array(test, dtype=dt)
        trainset = f.create_dataset('train', data=trainset)
        testset = f.create_dataset('test', data=testset)


data_process()
