from torchvision.datasets import MNIST, EMNIST, utils
import requests
import shutil
import os.path
import h5py
import numpy as np
import tarfile
import bz2


def data_process():

    path = './data_in_use/shakespeare/'
    url = 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+'shakespeare.tar.bz2'):
        res = requests.get(url)
        with open(path+'shakespeare.tar.bz2', mode='wb') as f:  # 需要用wb模式
            f.write(res.content)
        archive = tarfile.open(path+'shakespeare.tar.bz2', 'r:bz2')
        archive.debug = 1    # Display the files beeing decompressed.
        for tarinfo in archive:
            archive.extract(tarinfo, path)
        archive.close()

    with h5py.File(path+'shakespeare_train.h5', 'r') as read_train, \
         h5py.File(path+'shakespeare_test.h5', 'r') as read_test, \
         h5py.File('./data_in_use/shakespeare.h5', 'w') as fw:
        
        # 处理训练数据
        data = read_train['examples']
        train = []
        for k, v in enumerate(data.values()):
            snippets = list(v['snippets'][:])  # 假设snippets以字节方式存储在HDF5中
            user_id = [k+1]*len(snippets)  # 根据枚举生成用户ID
            train += list(zip(user_id, snippets))

        # 处理测试数据
        data = read_test['examples']
        test = []
        for k, v in enumerate(data.values()):
            snippets = list(v['snippets'][:])
            user_id = [k+1]*len(snippets)
            test += list(zip(user_id, snippets))

        # 定义结构化数组的数据类型
        dt = {'names': ['user_id', 'snippets'],
              'formats': ['int64', 'S726']}  # 根据数据需求调整字符串长度

        # 创建numpy结构化数组
        trainset = np.array(train, dtype=dt)
        testset = np.array(test, dtype=dt)

        # 在HDF5文件中保存数据集
        trainset = fw.create_dataset('train', data=trainset)
        testset = fw.create_dataset('test', data=testset)

        a = 1


data_process()
