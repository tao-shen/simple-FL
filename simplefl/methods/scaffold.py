from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class Scaffold(FedAvgM):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.c = copy.deepcopy(self.server.model)
        self.zero_weights(self.server.c)


    def clients_init(self):
        for client in self.clients:
            client.c = copy.deepcopy(self.server.c)

    def server_update(self):
        lr_g = self.args.lr_g
        w_t = self.server.model.state_dict()
        c_t = self.server.c.state_dict()
        delta_w_t = self.averaging(self.delta_models)
        delta_c_t = self.averaging(self.delta_cs)
        w = {}
        c = {}
        for key in w_t.keys():
            w[key] = w_t[key]-lr_g*delta_w_t[key]
            c[key] = c_t[key]-lr_g*delta_c_t[key]
        self.server.model.load_state_dict(w)
        self.server.c.load_state_dict(c)

    def clients_update(self):
        self.delta_models = []
        self.delta_cs = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model, c_k = self.local_update(
                self.clients[k], self.server.model, self.server.c)
            # c_k = copy.deepcopy(self.clients[k].c)
            # c_k.load_state_dict(c)
            delta_model = self.diff_model(model, self.server.model)
            delta_c = self.diff_model(c_k, self.server.c)
            self.clients[k].c = c_k
            self.delta_models.append(delta_model)
            self.delta_cs.append(delta_c)

    def local_update(self, client, model_g, server_c):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l, copy.deepcopy(server_c)
        model_g = copy.deepcopy(model_g).to(self.args.device)
        c_g = copy.deepcopy(server_c).to(self.args.device).state_dict()
        c_i = copy.deepcopy(client.c).to(self.args.device).state_dict()
        for E in range(self.E):
            self.fit_scaffold(model_l, train_loader, optimizer, c_g, c_i)
        k = self.E*int(len(client.train_data)/self.args.local_batch_size+1)
        for key in c_i.keys():
            c_i[key] -= c_g[key]-(model_g.state_dict()[key] -
                                  model_l.state_dict()[key])/(k*self.args.lr_l)
        c = copy.deepcopy(client.c)
        c.load_state_dict(c_i)
        return model_l, c

    def fit_scaffold(self, model, train_loader, optimizer, c_g, c_i):
        model.train().to(model.device)
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc='local_update')

        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            output = model(batch)
            label = batch['label']
            loss = model.loss_fn(output, label)
            loss.backward()
            optimizer.step()
            model_dict = model.state_dict()
            for key in model_dict.keys():
                model_dict[key] -= self.args.lr_l*(c_g[key]-c_i[key])
            loss_total += loss.item()
            loss_avg = loss_total / (idx + 1)
        model.train_num = len(train_loader.dataset)
