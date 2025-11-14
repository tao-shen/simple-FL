from sklearn import ensemble
from utils import *
import random
import copy
# import dgl
from torch.utils.data import DataLoader
from models import *


class Server:
    '''
    data are collected from clients
    '''

    def __init__(self, init, args):
        # self.features = args.features
        self.args = args
        self.init=init
        self.train_data, self.test_data = init.data
        train_set = Dataset(self.train_data, args)
        test_set = Dataset(self.test_data, args)
        self.train_loader = DataLoader(
            train_set, batch_size=args.server_batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_set, batch_size=args.eval_batch_size)
    
    def init_proxy_data(self):
        print('loading proxy data...')
        labels = self.init.proxy_data['label']

        # # 确定类别数量
        # num_classes = len(np.unique(labels))

        # # 创建一个字典来存储每个类别的选择样本
        # ind = []

        # # 对于每个类别，选择一个样本
        # for class_label in range(num_classes):
        #     # 找到属于当前类别的样本索引
        #     class_indices = np.where(labels == class_label)[0]
            
        #     # 从当前类别的样本中选择一个样本
        #     selected_sample_index = ind.append(np.random.choice(class_indices))
        
        # self.init.proxy_data=self.init.proxy_data[ind]
        return self.init.proxy_data
        


    def init_model(self):
        if 'ml-1m' in self.args.dataset:
            model = DIN(self.args)
        elif self.args.dataset == 'cifar10':
            model = ResNet20(self.args, num_classes=10)
        elif self.args.dataset == 'cifar100':
            model = ResNet20(self.args, num_classes=100)
        elif 'femnist' in self.args.dataset:
            model = CNN_FEMNIST(self.args, num_classes=62)
        elif 'fashionmnist' in self.args.dataset:
            model = LeNet5(self.args)
        return model


class Client():
    def __init__(self, train_data, test_data, args):
        self.train_data = train_data
        self.test_data = test_data
        # self.train_data = Dataset(train_data,args)
        # self.test_data = Dataset(test_data,args)
        self.args = args

    def local_update(self, model, train_loader=None, test_loader=None, ifprint=False):
        model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            if train_loader == None:
                train_loader = DataLoader(
                    Dataset(self.train_data, self.args), batch_size=10000, shuffle=True)
            if test_loader == None:
                test_loader = DataLoader(
                    Dataset(self.test_data, self.args), batch_size=10000, shuffle=False)
        except:
            return model
        # model.to(self.args.device)
        if 'prox' in self.args.method:
            prox = copy.deepcopy(model).to(model.device).state_dict()
            for E in range(self.args.local_epochs):
                model.fit_prox(train_loader, optimizer, prox)
        else:
            for E in range(self.args.local_epochs):
                l = model.fit(train_loader, optimizer)
        # model.evaluate(test_loader, ifprint=ifprint)
        # for saving cuda!
        # self.optimizer = []
        # self.model.cpu()
        # print(self.profile['userID'], 'end')
        return model


def ltensor(x):
    data = {}
    for k, v in x.items():
        data[k] = torch.tensor(v)
    return data


def init_clients(init, args):
    train_data, test_data = init.data
    user_offset = init.user_offset
    idx_train = user_offset['train']
    idx_test = user_offset['test']
    clients = [Client(train_data[idx_train[i]:idx_train[i+1]],
                      test_data[idx_test[i]:idx_test[i+1]], args) for i in range(len(idx_train)-1)]
    # slow
    # clients2 = [Client(train_data[train_data['user_id']==i], test_data[test_data['user_id']==i], args) for i in range(num_clients)]
    return clients


# def to_feats(models):
#     feats = {}
#     for k in models[0].state_dict().keys():
#         dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
#         v = torch.cat(tuple(dicts), dim=0)
#         k = k.replace('.', '-')
#         feats[k] = v
#     return feats


def to_model(models):
    feats = {}
    for k in models[0].state_dict().keys():
        dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
        v = torch.cat(tuple(dicts), dim=0)
        k = k.replace('.', '-')
        feats[k] = v
    return feats


def to_feats(models):
    feats = {}
    for k in models[0].state_dict().keys():
        dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
        v = torch.cat(tuple(dicts), dim=0)
        k = k.replace('.', '-')
        feats[k] = v
    return feats


def load_model(model, model_dict):
    for k, v in model_dict.items():
        d = k.split('-')
        m = model
        for k in d[:-1]:
            m = m._modules[k]
        m._parameters[d[-1]] = v
        # m._parameters[d[-1]] = v.clone()
        # m._parameters[d[-1]].retain_grad()


def to_device(x, device):
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = value.to(device)
    elif isinstance(x, Dataset):
        x = to_device(x.data, device)
    else:
        x = x.to(device)
    return x


# def to_cuda(x, device):
#     if isinstance(x, dict):
#         for key, value in x.items():
#             x[key] = value.cuda()
#     elif isinstance(x, Dataset):
#         x = to_device(x.data, device)
#     else:
#         x = x.cuda()
#     return x
