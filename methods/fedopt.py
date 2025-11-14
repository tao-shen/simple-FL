from .fedavgm import FedAvgM
from .fl import *
from utils import *
import copy
import torch


class FedOpt(FedAvgM):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        super().server_init()
        # Initialize accumulators for FedAdagrad and FedYogi
        self.server.model = self.server.init_model()
        m = copy.deepcopy(self.server.model)
        self.zero_weights(m)
        self.m = m.state_dict()
        v = copy.deepcopy(self.server.model)
        self.zero_weights(v)
        self.v = v.state_dict()

    def server_update(self):
        lr_g = self.args.lr_g
        beta1 = self.args.beta1
        beta2 = self.args.beta2
        tau = self.args.tau
        w_t = self.server.model.state_dict()
        delta_w_t = self.averaging(self.delta_models)

        # Iterate over each key in model's state dictionary and apply updates
        for key in w_t.keys():
            w_t[key] = self.update_weights(key, w_t, delta_w_t, lr_g, beta1, beta2, tau)

        self.server.model.load_state_dict(w_t)

    def update_weights(self, key, w_t, delta_w_t, lr_g, beta1, beta2, tau, opt='FedAdam'):
        # Update logic based on optimizer type
        if opt == 'FedAdam':
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * delta_w_t[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * delta_w_t[key].pow(2)
            update_term = self.m[key] / (torch.sqrt(self.v[key]) + tau)
        elif opt == 'FedAdagrad':
            self.v[key] += delta_w_t[key].pow(2)
            update_term = delta_w_t[key] / (torch.sqrt(self.v[key]) + tau)
        elif opt == 'FedYogi':
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * delta_w_t[key]
            v_prev = self.v[key]
            self.v[key] -= (1 - beta2) * (delta_w_t[key].pow(2) * torch.sign(v_prev - delta_w_t[key].pow(2)))
            update_term = self.m[key] / (torch.sqrt(self.v[key]) + tau)
        else:
            raise ValueError("Unsupported optimizer type")

        # Apply the computed update term to the model weights
        return w_t[key] - lr_g * update_term
