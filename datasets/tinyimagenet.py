from torchvision.datasets import MNIST, EMNIST, utils
import requests
import shutil
import os.path
import h5py
import numpy as np
import tarfile
import zipfile
from PIL import Image

def process_tinyimagenet():
    path = './data_in_use/tinyimagenet/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_file_path = path + 'tiny-imagenet-200.zip'

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(zip_file_path):
        # 下载zip文件
        res = requests.get(url)
        with open(zip_file_path, 'wb') as f:
            f.write(res.content)
        # 解压zip文件
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    
    base_path = path + 'tiny-imagenet-200'
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')

    # 读取类标识符文件
    wnids_path = os.path.join(base_path, 'wnids.txt')
    with open(wnids_path, 'r') as f:
        wnids = [x.strip() for x in f.readlines()]

    # 构建wnid到索引的映射
    wnid_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

    train, val = [], []
    
    # 处理训练数据
    for class_folder_name in os.listdir(train_path):
        class_folder_path = os.path.join(train_path, class_folder_name, 'images')
        label = wnid_to_idx[class_folder_name]
        for image_file_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file_name)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image).reshape((3, 64, 64))
            train.append((image, label))

    # 处理验证数据
    with open(os.path.join(val_path, 'val_annotations.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_file_name = parts[0]
            label = wnid_to_idx[parts[1]]
            image_path = os.path.join(val_path, 'images', image_file_name)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image).reshape((3, 64, 64))
            val.append((image, label))
    
    dt = {'names': ['pixels', 'label'],
          'formats': ['(3,64,64)float32', 'int64']}
    trainset = np.array(train, dtype=dt)
    valset = np.array(val, dtype=dt)
    
    # 保存数据
    with h5py.File('./data_in_use/tinyimagenet.h5', 'w') as fw:
        fw.create_dataset('train', data=trainset)
        fw.create_dataset('test', data=valset)

process_tinyimagenet()
