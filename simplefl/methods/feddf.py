from simplefl.methods.fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F


class FedDF(FedAvg):

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
                with torch.no_grad():
                    ensemble = []
                    for model in self.models:
                        model.eval().to(self.server.model.device)
                        logits = model(batch)
                        ensemble.append(logits)
                avg_logits = torch.stack(ensemble).mean(dim=0)
                loss = self.kl_loss(pred, avg_logits, self.args)
                loss.backward()
                optimizer.step()
                print(E, loss.item())

    def kl_loss(self, pred, target, args):
        if args.dataset == 'femnist' or args.dataset == 'fashionmnist':
            loss = F.kl_div(torch.log_softmax(pred, dim=1),
                            F.softmax((target), dim=1))
        elif args.dataset == 'ml-1m' or args.dataset == 'ml-100k':
            loss = self.binary_kl(F.logsigmoid(pred), torch.sigmoid(target))
        return loss

    def binary_kl(self, input, target):
        input = torch.exp(input)
        loss = input*torch.log(input/target)+(1-input) * \
            torch.log((1-input)/(1-target))
        loss = torch.mean(loss)
        return loss
