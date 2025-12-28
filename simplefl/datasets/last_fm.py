import pandas as pd
import h5py, random
import numpy as np

def last_fm():
    url='https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
    # load users.dat
    u_names=['user_id','artist_id','weight']
    users = pd.read_table('~/data/last_fm/user_artists.dat')
    groups=users.groupby(by='userID')
    friends = pd.read_table('~/data/last_fm/user_friends.dat')
    u = {'userID': [], 'ipv_item_seq': [], 'weight_seq': []}
    for i, user in groups:
        artistID = user['artistID'].values
        weight = user['weight'].values
        if len(artistID)<50:
            artistID = np.pad(artistID, (0, 50-len(artistID)), constant_values=0)
            weight = np.pad(weight, (0, 50-len(weight)), constant_values=0)
        u['userID'].append(i)
        u['ipv_item_seq'].append(artistID)
        u['weight_seq'].append(weight)
    users = pd.DataFrame(u)
    # load ratings.dat
    r_names = ['user_id', 'artist_id', 'tag_id', 'timestamp']
    ratings = pd.read_table('~/data/last_fm/user_taggedartists-timestamps.dat')
    ratings=ratings.groupby(['userID','artistID']).size().reset_index(name='Freq')
    # merge on 'user_id'
    history = pd.merge(users,ratings,on='userID')
    # # mapping
    for k in ['userID']:
        array=history[k].unique()
        m= {array[j-1]:j for j in range(1,len(array)+1)}
        history[k]=history[k].map(m)
        friends['userID'] = friends['userID'].map(m)
        friends['friendID'] = friends['friendID'].map(m)
    # generate labels
    labels = history['Freq'] >= 0
    history['label']= labels.values.astype('int32')
    # sort samples
    # history.sort_values(by=['user_id', 'timestamp'],inplace=True)
    # history.index = history['user_id'].values
    history=history.rename(columns={'artistID': 'cand_item_id', 'userID':'user_id'})
    count = history['cand_item_id'].value_counts()
    bins = pd.cut(count, 5)
    item_set = set(range(1, max(history['cand_item_id'])+1))
    # process samples for each user
    groups = history.groupby(by='user_id')
    # save as hdf5
    with h5py.File('last_fm.h5', 'w') as f:
        # customize dtype
        names = list(history.columns)
        formats = ['int32'] + ['50int32']*2+['int32']*3
        # formats.insert(5, '50int32')
        dt = {'names': names, 'formats': formats}
        # create datasets
        trainset, testset=[],[]
        # add each user in datasets
        train_off, test_off = [0],[0]

        # negtive items
        vlen = {'names':['train','test'], 'formats':[h5py.vlen_dtype('int32')]*2}
        user_neg_item= f.create_dataset('user_neg_item',(len(groups),),dtype=vlen)
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
            # try:
            #     neg_train = np.hstack(neg_train)
            # except:
            #     neg_train = np.array(neg_train)
            # neg_test=[]
            # for s in seq[split:]:
            #     neg_item = bins[bins == bins[s]].index.values
            #     neg_test.append(np.random.choice(neg_item, 99))
            # neg_test = np.hstack(neg_test)
            user_neg_item[i-1] = (neg_train, neg_test)
            # generate item sequence
            # seqs, cand_items = [], []
            # for j in range(num_samples):
            #     his = [0]*50+seq[:j]
            #     seqs+=[his[-50:]]
            # user.insert(5, 'ipv_item_seq', seqs)
            utrain = user[:split].to_numpy()
            utest = user[split:].to_numpy()
            train_off.append(train_off[-1]+split)
            test_off.append(test_off[-1]+num_samples-split)
            trainset += list(map(tuple, utrain))
            testset += list(map(tuple, utest))
        trainset = np.array(trainset,dtype=dt)
        testset = np.array(testset, dtype=dt)
        trainset = f.create_dataset('train', data=trainset)
        testset = f.create_dataset('test', data=testset)
        dt = {'names': ['train','test'], 'formats': [str(len(groups)+1)+'int32']*2}
        user_offset= f.create_dataset('user_offset', data=np.array((train_off, test_off), dtype=dt))
        # user_neg_items= f.create_dataset('user_neg_items', data=np.array((train_off, test_off), dtype=dt))
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

last_fm()




