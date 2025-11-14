from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedAvg(FL):

    def __init__(self, server, clients, args):
        self.server = server
        self.clients = clients
        self.args = args
        self.N = len(self.clients)
        self.T = args.communication_rounds
        self.C = args.client_fraction
        self.E = args.local_epochs
        if args.clients_per_round is None:
            self.K = int(self.C*self.N)
        else:
            self.K = args.clients_per_round
        self.opts = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}

    def server_init(self):
        self.server.model = self.server.init_model()

    def candidates_sampling(self):
        self.candidates = np.random.choice(
            np.arange(self.N), self.K, replace=False)

    def clients_update(self):
        self.models = []
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
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        for E in range(self.E):
            model_l.fit(train_loader, optimizer)
        return model_l

    def averaging(self, models):
        with torch.no_grad():
            model = copy.deepcopy(models[0])
            w = model.state_dict()
            num_sum = sum([m.train_num for m in models])
            for key in w.keys():
                w[key] = 0
                if self.args.avg == 'w':
                    for m in models:
                        w[key] += m.state_dict()[key]*m.train_num
                    w[key] /= num_sum
                else:
                    for m in models:
                        w[key] += m.state_dict()[key]
                    w[key] /= len(models)
            # model.load_state_dict(w)
        return w

    def evaluate(self):
        self.server.model.evaluate(
            self.server.test_loader, user_offset=self.server.init.user_offset['test'], recorder=self.args.recorder, ifprint=True)
