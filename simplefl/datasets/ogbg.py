from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader

dataset = DglGraphPropPredDataset(name='ogbg-molhiv', root='data')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(
    dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=collate_dgl)
valid_loader = DataLoader(
    dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)

def data_process():
    path = './data/'
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
