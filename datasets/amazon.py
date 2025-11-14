import random
import h5py
import numpy as np
import pandas as pd
# from utils import *
from sklearn.utils import shuffle
import os
import requests

## Amazon Video Games


def amazon():
    path = './data_in_use/amazon/'
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv'

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+'ratings_Video_Games.csv'):
        res = requests.get(url)
        with open(path+'ratings_Video_Games.csv', mode='wb') as f:  # 需要用wb模式
            f.write(res.content)
    # load csv
    names = ['user_id', 'cand_item_id', 'rating', 'timestamp']
    history = pd.read_table(
        path+'ratings_Video_Games.csv', sep=',', names=names)
    # mapping
    for k in ['user_id', 'cand_item_id']:
        array = history[k].unique()
        m = {array[j-1]: j for j in range(1, len(array)+1)}
        history[k] = history[k].map(m)
    # generate labels
    labels = history['rating'] >= 4
    history['label'] = labels.values.astype('float32')
    # sort samples
    history.sort_values(by=['user_id', 'timestamp'], inplace=True)
    # history.index = history['user_id'].values
    count = history['cand_item_id'].value_counts()
    # bins = pd.qcut(count, 5)
    item_set = set(range(1, max(history['cand_item_id'])+1))
    # process samples for each user
    groups = history.groupby(by='user_id')
    # save as hdf5
    with h5py.File('./data_in_use/amazon.h5', 'w') as f:
        # customize dtype
        names = list(history.columns)
        names.insert(names.index('cand_item_id'), 'ipv_item_seq')
        formats = ['int32'] * (len(history.columns)-1) + ['float32']
        formats.insert(names.index('cand_item_id')-1, '50int32')
        dt = {'names': names, 'formats': formats}
        # create datasets
        trainset, testset = [], []
        # add each user in datasets
        train_off, test_off = [0], [0]

        # negtive items
        vlen = {'names': ['train', 'test'],
                'formats': [h5py.vlen_dtype('int32')]*2}
        user_neg_item = f.create_dataset(
            'user_neg_item', (len(groups),), dtype=vlen)
        for i, user in groups:
            seq = user['cand_item_id'].tolist()
            samples = user.to_numpy()
            num_samples = len(samples)
            # split = int(num_samples*0.8)
            split = num_samples-1
            neg_items = np.array(list(item_set-set(seq)))
            neg_train = np.random.choice(neg_items, 4*len(seq[:split]))
            neg_test = np.random.choice(neg_items, 99*len(seq[split:]))
            # neg_train=[]
            # for s in seq[:split]:
            #     neg_item = bins[bins == bins[s]].index.values
            #     neg_train.append(np.random.choice(neg_item, 4))
            # neg_train = np.hstack(neg_train)
            # neg_test=[]
            # for s in seq[split:]:
            #     neg_item = bins[bins == bins[s]].index.values
            #     neg_test.append(np.random.choice(neg_item, 99))
            # neg_test = np.hstack(neg_test)
            user_neg_item[i-1] = (neg_train, neg_test)
            # generate item sequence
            seqs, cand_items = [], []
            for j in range(num_samples):
                his = [0]*50+seq[:j]
                seqs += [his[-50:]]
            user.insert(names.index('cand_item_id')-1, 'ipv_item_seq', seqs)
            utrain = user[:split].to_numpy()
            utest = user[split:].to_numpy()
            train_off.append(train_off[-1]+split)
            test_off.append(test_off[-1]+num_samples-split)
            trainset += list(map(tuple, utrain))
            testset += list(map(tuple, utest))
        trainset = np.array(trainset, dtype=dt)
        testset = np.array(testset, dtype=dt)
        trainset = f.create_dataset('train', data=trainset)
        testset = f.create_dataset('test', data=testset)
        dt = {'names': ['train', 'test'], 'formats': [
            str(len(groups)+1)+'int32']*2}
        user_offset = f.create_dataset(
            'user_offset', data=np.array((train_off, test_off), dtype=dt))


amazon()
