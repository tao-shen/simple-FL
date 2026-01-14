from torchvision.datasets import MNIST, EMNIST, utils
import requests
import shutil
import os.path
import h5py
import numpy as np
import tarfile
import bz2


def data_process():

    path = './data/stack_overflow/'
    url = 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+'fed_emnist.tar.bz2'):
        res = requests.get(url)
        with open(path+'fed_emnist.tar.bz2', mode='wb') as f:  # 需要用wb模式
            f.write(res.content)
        archive = tarfile.open(path+'fed_emnist.tar.bz2', 'r:bz2')
        archive.debug = 1    # Display the files beeing decompressed.
        for tarinfo in archive:
            archive.extract(tarinfo, path)
        archive.close()

    with h5py.File(path+'fed_emnist_train.h5', 'r') as read_train, h5py.File(path+'fed_emnist_test.h5', 'r') as read_test, h5py.File('./data/femnist.h5', 'w') as fw:
        train, test = [], []

        data = read_train['examples']
        for k, v in enumerate(data.values()):
            images = list(1-np.expand_dims(v['pixels'][:], axis=1))
            labels = list(v['label'][:])
            u = [k+1]*len(labels)
            train += list(map(tuple, zip(u, images, labels)))

        data = read_test['examples']
        for k, v in enumerate(data.values()):
            images = list(1-np.expand_dims(v['pixels'][:], axis=1))
            labels = list(v['label'][:])
            u = [k+1]*len(labels)
            test += list(map(tuple, zip(u, images, labels)))

        dt = {'names': ['user_id', 'pixels', 'label'],
              'formats': ['int64', '(1,28,28)float32', 'int64']}
        trainset = np.array(train, dtype=dt)
        testset = np.array(test, dtype=dt)
        trainset = fw.create_dataset('train', data=trainset)
        testset = fw.create_dataset('test', data=testset)
        a = 1


data_process()
