from .fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedRecom(FedAvg):
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
    
    def server_init(self):
        super().server_init()
        self.model_pool=[copy.deepcopy(self.server.model.state_dict()) for _ in range(self.K)]

    def server_update(self):
        model_dicts=[model.state_dict() for model in self.models]
        for key in self.server.model.state_dict().keys():
            params=[model[key] for model in model_dicts]
            np.random.shuffle(params)
            for i in range(len(model_dicts)):
                model_dicts[i][key]=params[i]        
        self.model_pool=model_dicts
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)
        
        
    def clients_update(self):
        self.models = []        
        for n, k in tqdm(enumerate(self.candidates), desc=self.__class__.__name__):
            recom=copy.deepcopy(self.server.model)
            recom.load_state_dict(self.model_pool[n])
            model = self.local_update(self.clients[k], recom)
            self.models.append(model.cpu())
            
    def evaluate(self):
        self.server.model.evaluate(
            self.server.test_loader, user_offset=self.server.init.user_offset['test'], recorder=self.args.recorder, ifprint=True)