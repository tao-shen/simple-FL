from dataclasses import replace
import pandas as pd
import h5py
import numpy as np

def string2array(str):
    return np.fromstring(str, dtype='int32', sep=',')
    

def mcc():
    # load dataset
    train = pd.read_csv('~/train_v4.csv')
    test = pd.read_csv('~/test_v4.csv')
    for k, v in train.items():
        if 'seq' in k and 'length' not in k:        
            train[k]=v.map(string2array)
    for k, v in test.items():
        if 'seq' in k and 'length' not in k:
            test[k] = v.map(string2array)
    groups_train = train.groupby(by='user_id')
    groups_test = test.groupby(by='user_id')
    # save as hdf5
    with h5py.File('mcc_v4.h5', 'w') as f:
        # customize dtype
        names = list(train.keys())
        # formats = ['int32']*12+['100int32']*6+['50int32']*6+['int32']+['float32']
        formats = ['int32']*12+['100int32']*6+['50int32']*6+['float32']+['int32']
        dt = {'names': names, 'formats': formats}
        trainset, testset = [], []
        # add each user in datasets
        train_off, test_off = [0], [0]
        for _, user in groups_train:
            train_off.append(train_off[-1]+len(user))
            utrain = list(user.to_numpy())
            trainset += list(map(tuple, utrain))
        for _, user in groups_test:
            test_off.append(test_off[-1]+len(user))
            utest = list(user.to_numpy())
            testset += list(map(tuple, utest))
        trainset = np.array(trainset, dtype=dt)
        testset = np.array(testset, dtype=dt)
        trainset = f.create_dataset('train', data=trainset)
        testset = f.create_dataset('test', data=testset)
        dt = {'names': ['train', 'test'], 'formats': [
            str(len(groups_train)+1)+'int32', str(len(groups_test)+1)+'int32']}
        user_offset = f.create_dataset(
            'user_offset', data=np.array((train_off, test_off), dtype=dt))
        # trainset.attrs['user_offset'] = np.array(train_off)
        # testset.attrs['user_offset'] = np.array(test_off)
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


mcc()
