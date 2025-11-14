from .fl import *
from .fedavg import FedAvg
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
from server_client import init_clients, Client
from data import dirichlet_split_noniid
import re


class FedAvg_Client_Drift_Only(FedAvg):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        self.server.model = self.server.init_model()
        self.iid_each_round()

    def clients_update(self):
        self.models = []
        self.noniid_each_client()
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())

    def server_update(self):
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.use_this_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        for E in range(self.E):
            model_l.fit(train_loader, optimizer)
        return model_l

    def iid_each_round(self):
        train_data, test_data = self.server.init.data
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        N_CLIENTS = len(np.unique(train_data['user_id']))
        user_offset = {'train': np.linspace(0, len(train_data), N_CLIENTS+1, endpoint=True).astype(int),
                       'test': np.linspace(0, len(test_data), N_CLIENTS+1, endpoint=True).astype(int)}
        idx_train = user_offset['train']
        idx_test = user_offset['test']
        self.clients = [Client(train_data[idx_train[i]:idx_train[i+1]],
                               test_data[idx_test[i]:idx_test[i+1]], self.args) for i in range(len(idx_train)-1)]

    def noniid_each_client(self):
        data = np.concatenate(
            [self.clients[i].train_data for i in self.candidates])
        np.random.shuffle(data)
        labels = data['label']
        N_CLIENTS = len(self.candidates)
        DIRICHLET_ALPHA = float(re.findall(r"\d+\.?\d*", self.args.iid)[0])
        idcs = dirichlet_split_noniid(
            labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

        for i, idx in zip(self.candidates, idcs):
            self.clients[i].use_this_data = data[idx]
