from .fedavgm import FedAvgM
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
from models.lstm import CoordinateLSTMOptimizer



class FedLeo(FedAvgM):

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()
        self.input_size=2
        self.hidden_size=10
        self.output_size=1
        self.aggr = CoordinateLSTMOptimizer(self.input_size, self.hidden_size, self.output_size, self.args).to(self.args.device)
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.AdamW}
        self.aggr_optimizer = opts[self.args.server_optimizer](
            self.aggr.parameters(),
            lr=0.01,
            # weight_decay=self.args.weight_decay,
        )
        self.proxy_loss = 0
        self.t = 0
        self.trunced_t=3
        v_m=self.server.model.to_vector().data.to(self.args.device)
        h0=torch.zeros(len(v_m),self.hidden_size, dtype=torch.float).to(self.args.device)
        h1=v_m.unsqueeze(dim=-1).expand_as(h0)
        self.hx = [h0,h1]
        # self.hx=None

    def server_update(self):
        # d=[dict(delta_model.state_dict()) for delta_model in self.delta_models]
        # b=dict(self.server.model.state_dict())
        deltas = [delta_model.to_vector() for delta_model in self.delta_models]
        deltas = torch.stack(deltas).data.T.to(self.args.device)
        # self.delta_models = to_device(self.delta_models, self.args.device)
        vec, self.hx = self.aggr(deltas, self.hx)
        # delta_parameters, self.hx = self.aggr(deltas, self.hx)
        # 取data使is_leaf=true, requires_grad=false
        # vec = self.server.model.to_vector().data.to(self.args.device)
        # vec = vec - delta_parameters
        proxy_model = copy.deepcopy(self.server.model)
        proxy_model_from_vector(proxy_model, vec)  # 这句执行后，模型参数不是leaf
        self.proxy_loss += self.eval_proxy_loss(proxy_model, self.server.proxy_data)
        self.server.model.from_vector(vec)  # 这句执行后，server.model里的参数还是leaf
        if self.t != 0 and self.t % self.trunced_t == 0:
            try:
                self.aggr_optimizer.zero_grad()
                self.proxy_loss/=self.trunced_t
                self.proxy_loss.backward()
                self.aggr_optimizer.step()
                self.proxy_loss = 0
                self.hx[0] = self.hx[0].data
                self.hx[1] = self.hx[1].data
            except:
                pass
        self.t += 1

    def eval_proxy_loss(self, proxy_model, proxy_data):
        loss = 0
        loss_fun = proxy_model.loss_fn
        proxy_set = Dataset(proxy_data, self.args)
        train_loader = DataLoader(
            proxy_set, batch_size=self.args.server_batch_size, shuffle=True
        )
        for E in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader, start=1):
                batch = to_device(batch, self.args.device)
                pred = proxy_model(batch)
                label = batch["label"]
                loss += (loss_fun(pred, label)-loss)/idx
                print(E, loss.data)
        return loss

    def aggr_update(self, feats):
        self.aggr.to(self.args.device).train()
        loss_fun = self.server.model.loss_fn
        proxy_set = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            proxy_set, batch_size=self.args.server_batch_size, shuffle=True
        )
        feats = to_device(feats, self.args.device)
        for E in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader):
                batch = to_device(batch, self.args.device)
                self.aggr_optimizer.zero_grad()
                model = self.aggr(feats)
                pred = model(batch)
                label = batch["label"]
                loss = loss_fun(pred, label)
                try:
                    loss.backward()
                except:
                    pass

                self.aggr_optimizer.step()
                print(E, loss.data)
            # self.server.model=

    def to_feats(self, models):
        feats = {}
        for k in models[0].state_dict().keys():
            dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
            v = torch.cat(tuple(dicts), dim=0)
            k = k.replace(".", "-")
            feats[k] = v
        return feats



def proxy_model_from_vector(model, vec):
    pointer = 0
    for name, _ in model.named_parameters():
        num_param = _.numel()
        splited_name = name.split(".")
        module = model
        param_name = splited_name[-1]
        for module_name in splited_name[:-1]:
            module = module._modules[module_name]
        param = vec[pointer : pointer + num_param].view_as(_)
        module._parameters[param_name] = param
        pointer += num_param


