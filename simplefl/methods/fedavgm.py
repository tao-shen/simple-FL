from .fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedAvgM(FedAvg):
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        self.server.model = self.server.init_model()
        m = copy.deepcopy(self.server.model)
        self.zero_weights(m)
        self.m = m.state_dict()

    def clients_update(self):
        self.delta_models = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            delta_model = self.diff_model(model, self.server.model)
            self.delta_models.append(delta_model)

    def server_update(self):
        lr_g = self.args.lr_g
        beta1 = self.args.beta1
        w_t = self.server.model.state_dict()
        delta_w_t = self.averaging(self.delta_models)
        w = {}
        for key in w_t.keys():
            self.m[key] = beta1*self.m[key]+(1-beta1)*delta_w_t[key]
            w[key] = w_t[key]-lr_g*self.m[key]
        self.server.model.load_state_dict(w)

    def diff_model(self, model_l, model_g):
        model_l.cpu()
        w_l = model_l.state_dict()
        w_g = model_g.state_dict()
        for key in w_l.keys():
            w_l[key] *= -1
            w_l[key] += w_g[key]
        return model_l

    def zero_weights(self, model):
        for n, p in model.named_parameters():
            p.data.zero_()
