from .fl import *
from .fedavg import FedAvg
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedAvg_Period_Drift_Only(FedAvg):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def clients_update(self):
        self.models = []
        self.iid_each_client()
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

    def iid_each_client(self):
        data = np.concatenate(
            [self.clients[i].train_data for i in self.candidates])
        np.random.shuffle(data)
        num_clients = len(self.candidates)
        idx = np.linspace(0, len(data), num_clients +
                          1, endpoint=True).astype(int)
        for i, id in zip(self.candidates, range(len(self.candidates))):
            self.clients[i].use_this_data = data[idx[id]:idx[id+1]]
