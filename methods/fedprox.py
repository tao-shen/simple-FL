from methods.fedavg import FedAvg
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedProx(FedAvg):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        prox = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        for E in range(self.E):
            self.fit_prox(model_l, train_loader, optimizer, prox)
        return model_l

    def fit_prox(self, model, train_loader, optimizer, prox):
        mu = self.args.mu
        model.train().to(model.device)
        prox = prox.to(model.device).state_dict()
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc='local_update')
        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            output = model(batch)
            label = batch['label']
            loss = model.loss_fn(output, label)
            prox_loss = 0.0
            for n, p in model.named_parameters():
                prox_loss += torch.square(prox[n]-p).sum()
            loss += mu/2*prox_loss
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            loss_avg = loss_total / (idx + 1)
        model.train_num = len(train_loader.dataset)
