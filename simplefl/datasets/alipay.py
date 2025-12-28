# %%
from sklearn.utils import shuffle
from utils import *
import pandas as pd
import h5py, random
import numpy as np


def alipay():
    # load alipay.txt
    names = ['user_id', 'cand_item_id', 'timestamp', 'label']
    history = pd.read_table('~/data/alipay/alipay.txt', sep=',', names=names)
    # sort samples
    history.sort_values(by=['user_id', 'timestamp'], inplace=True)
    # process samples for each user
    groups = history.groupby(by='user_id')
    count = history['cand_item_id'].value_counts()
    bins = pd.qcut(count, 5)
    # save as hdf5
    with h5py.File('alipay.h5', 'w') as f:
        # customize dtype
        names = list(history.columns)
        names.insert(1, 'ipv_item_seq')
        formats = ['int32'] * 4
        formats.insert(1, '50int32')
        dt = {'names': names, 'formats': formats}
        trainset, testset = [], []
        # add each user in datasets
        train_off, test_off = [0], [0]
        item_set = set(range(1, max(history['cand_item_id'])+1))
        # negtive items
        # vlen = {'names': ['train', 'test'],
        #         'formats': [h5py.vlen_dtype('int32')]*2}
        # user_neg_item = f.create_dataset(
        #     'user_neg_item', (len(groups),), dtype=vlen)
        for i, user in groups:
            seq = user['cand_item_id'].tolist()
            samples = user.to_numpy()
            num_samples = len(samples)
            split = int(num_samples*0.8)
            # split = num_samples-1
            # neg_items = np.array(list(item_set-set(seq)))
            # neg_train = np.random.choice(neg_items, 4*len(seq[:split]))
            # neg_test = np.random.choice(neg_items, 99*len(seq[split:]))
            # neg_train = []
            # for s in seq[:split]:
            #     neg_item = bins[bins == bins[s]].index.values
            #     neg_train.append(np.random.choice(neg_item, 4))
            # neg_train = np.hstack(neg_train)
            # neg_test = []
            # for s in seq[split:]:
            #     neg_item = bins[bins == bins[s]].index.values
            #     neg_test.append(np.random.choice(neg_item, 99))
            # neg_test = np.hstack(neg_test)
            # user_neg_item[i-1] = (neg_train, neg_test)
            # generate item sequence
            seqs, cand_items = [], []
            for j in range(num_samples):
                his = [0]*50+seq[:j]
                seqs += [his[-50:]]
            user.insert(1, 'ipv_item_seq', seqs)
            # # train_samples 1:4
            # cand_items, labels = [],[]
            # for j in seq[:split]:
            #     cand = [j]+random.sample(neg_item, 4)
            #     label= [1]+[0]*4
            #     cand_items += cand
            #     labels += label
            # utrain=np.repeat(user[:split].to_numpy(),5,axis=0)
            # utrain[:, 6] = cand_items
            # utrain[:, 9] = labels
            # # test_samples 1:99
            # cand_items, labels = [], []
            # for j in seq[split:]:
            #     cand = [j]+random.sample(neg_item, 99)
            #     label = [1]+[0]*99
            #     cand_items += cand
            #     labels += label
            # utest = np.repeat(user[split:].to_numpy(), 100, axis=0)
            # utest[:, 6] = cand_items
            # utest[:, 9] = labels
            # train_off.append(train_off[-1]+split*5)
            # test_off.append(test_off[-1]+(num_samples-split)*100)
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
        dt = {'names': ['train','test'], 'formats': [str(len(groups)+1)+'int32']*2}
        user_offset = f.create_dataset('user_offset', data=np.array((train_off, test_off), dtype=dt))
        # f.attrs['read_me'] = 'Seqs should be reshaped as (-1,50)'

    # save dataset
    # ----------------------------------------
    # # save pd as pkl
    # history.to_pickle('ml-1m_full.pkl')
    # history[:10000].to_pickle('ml-1m_demo.pkl')
    # ----------------------------------------
    # # save users as pkl
    # with open('users_demo.pkl', 'wb') as f:
    #     pickle.dump({'train': data['train'][:100],
    #                 'test': data['test'][:100]}, f)
    # with open('users_full.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # ----------------------------------------
    # save pd as h5
    # history[:10000].to_hdf('ml-1m.h5','demo')
    # history.to_hdf('ml-1m.h5','full')
    # ----------------------------------------
    # # save users as h5 (as above)


alipay()
