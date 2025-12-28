from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedEve(FedAvgM):
    """
    In summary, the code implements the FedEve for Federated Learning(FL). It extends the `FedAvgM` class, which itself extends the `FL` class. The `server_init()` method initializes the server variables, while the `clients_update()` method updates the client models. The `server_update()` method updates the server model. The `r()` method calculates the R value, while the `q()` method calculates the Q value. 
    """

    def server_init(self):
        # Initialize server variables
        self.server.model = self.server.init_model()  # Initialize server model
        m = copy.deepcopy(self.server.model)
        self.zero_weights(m)  # Zero-out weights
        self.m = m.state_dict()  # Store the state dict of the zeroed-out model
        self.sigma = 0  # Initialize the covariance matrix as 0
        self.args.recorder['kal_Q'] = []  # Initialize recording of Q values
        self.args.recorder['kal_R'] = []  # Initialize recording of R values
        # Initialize recording of sigma values
        self.args.recorder['kal_sigma'] = []
        # Initialize recording of Kalman gain values
        self.args.recorder['kal_K'] = []

    def clients_update(self):
        # Update client models
        self.delta_models = []  # Initialize list of delta models
        self.w_predict = {}
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # Update the local model for each client
            # Make a copy of the server model
            predict = copy.deepcopy(self.server.model)
            # Load the predicted weight
            predict.load_state_dict(self.w_predict)
            # Perform the local update
            model = self.local_update(self.clients[k], predict)
            # Get the delta model
            delta_model = self.diff_model(model, predict)
            # Append the delta model to the list
            self.delta_models.append(delta_model)

    def server_update(self):
        # Update the server model
        w_t = self.server.model.state_dict()  # Get the current server weights
        # Get the averaged delta model
        delta_w_t = self.averaging(self.delta_models)
        self.R = self.r()  # Get the R value
        self.Q = self.q(delta_w_t)  # Get the Q value
        lr_g = self.args.lr_g  # Get the learning rate
        # Update the weights
        w = {}
        self.sigma = self.sigma + self.Q
        self.k = self.sigma/(self.sigma+self.R)
        self.sigma = (1-self.k)*self.sigma
        self.args.recorder['kal_Q'] += [self.Q.item()]
        self.args.recorder['kal_sigma'] += [self.sigma.item()]
        self.args.recorder['kal_R'] += [self.R.item()]
        self.args.recorder['kal_K'] += [self.k.item()]
        for key in w_t.keys():
            self.m[key] = self.m[key]+self.k * (delta_w_t[key]-self.m[key])
            w[key] = w_t[key]-lr_g*self.m[key]
        self.server.model.load_state_dict(w)  # Update the server model

    def r(self):
        with torch.no_grad():
            model = copy.deepcopy(self.delta_models[0])
            w = model.state_dict()
            for key in w.keys():
                w[key] = 0
                for m in self.delta_models:
                    w[key] += m.state_dict()[key]
                w[key] /= len(self.delta_models)
                # Calculate R value
                r = 0
                p1 = torch.hstack([p.flatten() for p in w.values()])
                for m in self.delta_models:
                    p2 = torch.hstack([p.flatten() for p in m.parameters()])
                    r += (p1-p2)**2
                r /= len(self.delta_models)**2
                r = torch.mean(r)
        return r

    def q(self, delta_w_t):
        p2 = torch.hstack([p.flatten() for p in delta_w_t.values()])
        q = torch.mean((p2) ** 2/len(self.delta_models))
        return q
