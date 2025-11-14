from methods.fedavg import FedAvg
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F


class FedMeta(FedAvg):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()

    def server_update(self):
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)
        proxy_set = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            proxy_set, batch_size=self.args.server_batch_size, shuffle=True)
        optimizer = self.opts[self.args.server_optimizer](self.server.
                                                     model.parameters(), lr=self.args.lr_g, weight_decay=self.args.weight_decay)
        for E in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch = to_device(batch, self.args.device)
                self.server.model.to(self.server.model.device)
                pred = self.server.model(batch)
                label = batch['label']
                loss = self.server.model.loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                print(E, loss.item())
