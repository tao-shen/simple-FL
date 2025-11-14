from .fedavg import FedAvg
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class Mime(FedAvg):
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
        self.stats = None  # 用于存储优化统计信息
        self.base_optimizer = "sgdm"  # 设置基础优化器为 sgdm

    def server_init(self):
        self.server.model = self.server.init_model()
        self.stats = self.init_stats()  # 初始化优化统计信息

    def init_stats(self):
        # 根据基础算法初始化统计信息
        if self.base_optimizer == "sgd":
            return None
        elif self.base_optimizer == "sgdm":
            return {
                k: torch.zeros_like(v)
                for k, v in self.server.model.state_dict().items()
            }
        elif self.base_optimizer == "adam":
            return {
                "m": {
                    k: torch.zeros_like(v)
                    for k, v in self.server.model.state_dict().items()
                },
                "v": {
                    k: torch.zeros_like(v)
                    for k, v in self.server.model.state_dict().items()
                },
            }

    def clients_update(self):
        self.models = []
        self.gradients = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model, gradient = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())
            self.gradients.append(gradient)

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )

        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args),
                batch_size=self.args.local_batch_size,
                shuffle=True,
            )
        except:  # skip dataless client
            return model_l, None

        for E in range(self.E):
            self.fit_mimelite(model_l, train_loader, optimizer)

        full_gradient = self.compute_full_gradient(model_l, client.train_data)
        return model_l, full_gradient

    def fit_mimelite(self, model, train_loader, optimizer):
        model = model.to(self.args.device).train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            output = model(batch)
            loss = model.loss_fn(output, batch["label"])
            loss.backward()
            self.apply_stats(optimizer)  # 应用优化统计信息
            optimizer.step()

    def compute_full_gradient(self, model, data):
        model.eval()
        full_loader = DataLoader(
            Dataset(data, self.args), batch_size=len(data), shuffle=False
        )
        batch = next(iter(full_loader))
        batch = to_device(batch, model.device)
        output = model(batch)
        loss = model.loss_fn(output, batch["label"])
        loss.backward()
        return {name: param.grad.clone() for name, param in model.named_parameters()}

    def server_update(self):
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)
        self.update_stats()

    def average_gradients(self):
        return {
            k: torch.stack(
                [
                    g[k]
                    for g in self.gradients
                    if g is not None and k in g and g[k] is not None
                ]
            ).mean(0)
            for k in set().union(*[g.keys() for g in self.gradients if g is not None])
        }

    def apply_stats(self, optimizer):
        if self.base_optimizer == "sgd":
            return
        elif self.base_optimizer == "sgdm":
            for group in optimizer.param_groups:
                for name, p in zip(self.stats.keys(), group["params"]):
                    if p.grad is None:
                        continue
                    p.grad.add_(self.stats[name].to(p.device), alpha=group["momentum"])
        elif self.base_optimizer == "adam":
            for group in optimizer.param_groups:
                for name, p in zip(self.stats["m"].keys(), group["params"]):
                    if p.grad is None:
                        continue
                    p.grad.mul_(1 - group["betas"][0]).add_(
                        self.stats["m"][name], alpha=group["betas"][0]
                    )
                    p.grad.mul_(1 / (torch.sqrt(self.stats["v"][name]) + group["eps"]))

    def update_stats(self):
        if self.base_optimizer == "sgd":
            return
        elif self.base_optimizer == "sgdm":
            beta = self.args.beta1
            g_avg = self.average_gradients()
            for k in self.stats.keys():
                self.stats[k] = beta * self.stats[k] + (1 - beta) * g_avg[k].cpu()
        elif self.base_optimizer == "adam":
            beta1, beta2 = self.args.beta1, self.args.beta2
            g_avg = self.average_gradients()
            for k in self.stats["m"].keys():
                self.stats["m"][k] = beta1 * self.stats["m"][k] + (1 - beta1) * g_avg[k]
                self.stats["v"][k] = beta2 * self.stats["v"][k] + (1 - beta2) * (
                    g_avg[k] ** 2
                )
