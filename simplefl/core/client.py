"""
Client module for Simple-FL
"""

import torch
import copy
from torch.utils.data import DataLoader
from simplefl.utils.config import Dataset


class Client:
    """
    Federated Learning Client
    
    Each client has local training and test data.
    """

    def __init__(self, train_data, test_data, args):
        """
        Initialize client
        
        Args:
            train_data: Client's training data
            test_data: Client's test data
            args: Arguments containing client configuration
        """
        self.train_data = train_data
        self.test_data = test_data
        # self.train_data = Dataset(train_data,args)
        # self.test_data = Dataset(test_data,args)
        self.args = args

    def local_update(self, model, train_loader=None, test_loader=None, ifprint=False):
        """
        Perform local training on client
        
        Args:
            model: Global model to train locally
            train_loader: Optional custom training data loader
            test_loader: Optional custom test data loader
            ifprint: Whether to print evaluation results
            
        Returns:
            Locally trained model
        """
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


def init_clients(init, args):
    """
    Initialize all clients with their data partitions
    
    Args:
        init: Data initialization object
        args: Arguments containing configuration
        
    Returns:
        List of Client objects
    """
    train_data, test_data = init.data
    user_offset = init.user_offset
    idx_train = user_offset['train']
    idx_test = user_offset['test']
    clients = [Client(train_data[idx_train[i]:idx_train[i+1]],
                      test_data[idx_test[i]:idx_test[i+1]], args) for i in range(len(idx_train)-1)]
    # slow
    # clients2 = [Client(train_data[train_data['user_id']==i], test_data[test_data['user_id']==i], args) for i in range(num_clients)]
    return clients


def ltensor(x):
    """Convert dict values to tensors"""
    data = {}
    for k, v in x.items():
        data[k] = torch.tensor(v)
    return data


def to_model(models):
    """Convert list of models to feature dict"""
    feats = {}
    for k in models[0].state_dict().keys():
        dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
        v = torch.cat(tuple(dicts), dim=0)
        k = k.replace('.', '-')
        feats[k] = v
    return feats


def to_feats(models):
    """Convert list of models to feature dict"""
    feats = {}
    for k in models[0].state_dict().keys():
        dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
        v = torch.cat(tuple(dicts), dim=0)
        k = k.replace('.', '-')
        feats[k] = v
    return feats


def load_model(model, model_dict):
    """Load model parameters from dict"""
    for k, v in model_dict.items():
        d = k.split('-')
        m = model
        for k in d[:-1]:
            m = m._modules[k]
        m._parameters[d[-1]] = v
        # m._parameters[d[-1]] = v.clone()
        # m._parameters[d[-1]].retain_grad()
