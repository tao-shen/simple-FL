from .fedavgm import FedAvgM
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
# from models import *


class FedLeo(FedAvgM):

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()
        if self.args.dataset == 'femnist' or self.args.dataset == 'fashionmnist':
            norm = False
        elif self.args.dataset == 'ml-1m' or self.args.dataset == 'ml-100k':
            norm = True
        self.aggr = PA(self.server.model, self.args)
        opts = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
        self.aggr_optimizer = opts[self.args.server_optimizer](self.
                                                               aggr.parameters(), lr=self.args.lr_g, weight_decay=self.args.weight_decay)

    def server_update(self):
        feats = self.to_feats(self.delta_models)
        if 'agg1' in self.args.note:
            with torch.no_grad():
                self.aggr.to(self.args.device).eval()
                feats = to_device(feats, self.args.device)
                self.server.model = self.aggr(feats)
            self.aggr_update(feats)

        else:
            self.aggr_update(feats)
            with torch.no_grad():
                self.aggr.to(self.args.device).eval()
                feats = to_device(feats, self.args.device)
                self.server.model = self.aggr(feats)

    def aggr_update(self, feats):
        if self.args.dataset == 'femnist' or self.args.dataset == 'fashionmnist':
            alpha1 = 0
            alpha2 = 0
        elif self.args.dataset == 'ml-1m' or self.args.dataset == 'ml-100k':
            alpha1 = 1e-2
            alpha2 = 1e-2
        self.aggr.to(self.args.device).train()
        # optimizer = torch.optim.Adam(self.aggr.parameters(), lr=1e-2,
        #                              # momentum=0.5,
        #                              weight_decay=1e-4,
        #                              )
        loss_fun = self.server.model.loss_fn
        proxy_set = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            proxy_set, batch_size=self.args.server_batch_size, shuffle=True)
        feats = to_device(feats, self.args.device)

        # if 'g' in self.args.method:
        #     model = self.aggr(self.sg, feats)
        # else:
        # model = self.aggr(feats)
        for E in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader):
                batch = to_device(batch, self.args.device)
                self.aggr_optimizer.zero_grad()
                model = self.aggr(feats)
                # model_dict = {}
                # for k, v in h.items():
                #     # k = k.replace('-', '.')
                #     model_dict[k] = torch.mean(v, dim=0)
                # model = copy.deepcopy(self.model)
                # load_model(model, h)
                # proxy_set=next(iter(train_loader))
                # proxy_set = to_device(proxy_set, self.args.device)
                pred = model(batch)
                label = batch['label']
                loss = loss_fun(pred, label)

                # loss1, loss2 = 0.0, 0.0
                # for p1 in self.aggr.parameters():
                #     loss1 += torch.norm(p1, p=2)
                # for p in model.parameters():
                #     loss += 1e-5*torch.square(p).sum()
                # # for p1 in self.aggr.dx.values():
                # #     loss2 += torch.norm(p1, p=2)
                # loss += alpha1*loss1+alpha2*loss2
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
            k = k.replace('.', '-')
            feats[k] = v
        return feats


class PA(nn.Module):
    def __init__(self, init_model, args, norm=False):
        super(PA, self).__init__()
        self.args = args
        self.p = args.dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        self.p_weight = nn.ModuleDict()
        self.g_weight = nn.ModuleDict()
        self.b_weight = nn.ModuleDict()
        self.last_model = copy.deepcopy(init_model)
        self.last_params = {}
        self.meta_w_d = {}
        self.meta_w_lr = {}
        self.meta_bias = {}
        self.params = {}
        self.w_d = {}
        self.w_lr = {}
        self.bias = {}
        self.size = {}
        self.record = {'delta_w': [], 'u': []}
        for key, v in self.last_model.state_dict().items():
            size = v.size().numel()
            k = key.replace('.', '-')
            self.size[k] = v.size()
            hidden = int(torch.log2(torch.tensor(size))+1)
            self.p_weight[k] = nn.Sequential(
                nn.Linear(size, hidden, bias=False),
                self.activation,)
            self.g_weight[k] = nn.Sequential(
                nn.Linear(size, hidden, bias=False),
                self.activation)
            self.b_weight[k] = nn.Sequential(
                nn.Linear(2*hidden, size, bias=False),
                self.dropout,
            )
            self.last_params[k] = nn.Parameter(
                copy.deepcopy(v), requires_grad=False)
            self.register_parameter('last_model_'+k, self.last_params[k])

    def forward(self, x):
        h = {}
        for k in x.keys():
            self.params[k] = self.last_params[k].expand_as(x[k])
            p2h = self.p_weight[k](self.params[k].flatten(1))
            g2h = self.g_weight[k](x[k].flatten(1))
            self.bias[k] = self.b_weight[k](
                torch.cat((p2h, g2h), dim=-1)).view_as(x[k])
            # feats = torch.stack((x[k], self.params[k]), dim=-1)
            # self.bias[k] = self.weight[k](x[k].flatten(1)).squeeze(-1)
            if 'with_g' in self.args.note:
                self.params[k] = self.params[k] - \
                    x[k]*(1-self.bias[k]*(1-self.p))
            elif 'without_g' in self.args.note:
                self.params[k] = self.params[k] - \
                    x[k] + self.bias[k]*(1-self.p)
            # self.params[k] = self.params[k] - x[k]
            self.params[k] = self.params[k].mean(dim=0)

            # self.w_d[k] = self.w_d[k].mean(dim=0)
            # self.w_lr[k] = self.w_lr[k].mean(dim=0)
            # self.bias[k] = self.bias[k].mean(dim=0)
        model = copy.deepcopy(self.last_model)
        if self.training:
            load_model(model, self.params)
            return model
        else:
            w = {}
            for k, v in self.params.items():
                self.last_params[k] = v
                w[k.replace('-', '.')] = v
            model.load_state_dict(w)
            # self.record['delta_w'].append(x)
            # self.record['u'].append(copy.deepcopy(self.bias))
        return model

    def preprocess_gradients(x):
        p = torch.tensor(10)
        eps = 1e-6
        indicator = (x.abs() > torch.exp(-p)).float()
        x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
        x2 = x.sign() * indicator + torch.exp(p) * x * (1 - indicator)

        return torch.cat((x1, x2), 1)


def load_model(model, model_dict):
    for k, v in model_dict.items():
        d = k.split('-')
        m = model
        for k in d[:-1]:
            m = m._modules[k]
        m._parameters[d[-1]] = v


def model2vector(model):
    # Flattening the parameters
    param_shapes = {n: p.shape for n, p in model.named_parameters()}
    flattened_params = torch.hstack(
        [p.flatten() for p in model.parameters()])
    return flattened_params, param_shapes


def vector2model_dict(flattened_params, param_shapes):

    # Reshaping the flattened parameters
    start_idx = 0
    model_dict = {}
    for key, shape in param_shapes.items():
        size = np.prod(shape)
        model_dict[key] = flattened_params[start_idx:start_idx +
                                           size].reshape(shape)
        start_idx += size
    return model_dict


# class PA(nn.Module):
#     def __init__(self, init_model, args, norm=False):
#         super(PA, self).__init__()
#         self.args = args
#         self.p = args.dropout
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(self.p)
#         self.p_weight = nn.ModuleDict()
#         self.g_weight = nn.ModuleDict()
#         self.b_weight = nn.ModuleDict()
#         self.last_model = copy.deepcopy(init_model)
#         self.last_params = {}
#         self.meta_w_d = {}
#         self.meta_w_lr = {}
#         self.meta_bias = {}
#         self.params = {}
#         self.w_d = {}
#         self.w_lr = {}
#         self.bias = {}
#         self.size = {}
#         self.record = {'delta_w': [], 'u': []}
#         for key, v in self.last_model.state_dict().items():
#             size = v.size().numel()
#             k = key.replace('.', '-')
#             self.size[k] = v.size()
#             hidden = int(torch.log2(torch.tensor(size))+1)
#             self.p_weight[k] = nn.Sequential(
#                 nn.Linear(size, hidden, bias=False),
#                 self.activation,)
#             self.g_weight[k] = nn.Sequential(
#                 nn.Linear(size, hidden, bias=False),
#                 self.activation)
#             self.b_weight[k] = nn.Sequential(
#                 nn.Linear(2*hidden, size, bias=False),
#                 self.dropout,
#             )
#             self.last_params[k] = nn.Parameter(
#                 copy.deepcopy(v), requires_grad=False)
#             self.register_parameter('last_model_'+k, self.last_params[k])

#     def forward(self, x):
#         h = {}
#         for k in x.keys():
#             self.params[k] = self.last_params[k].expand_as(x[k])
#             p2h = self.p_weight[k](self.params[k].flatten(1))
#             g2h = self.g_weight[k](x[k].flatten(1))
#             self.bias[k] = self.b_weight[k](
#                 torch.cat((p2h, g2h), dim=-1)).view_as(x[k])
#             # feats = torch.stack((x[k], self.params[k]), dim=-1)
#             # self.bias[k] = self.weight[k](x[k].flatten(1)).squeeze(-1)
#             if 'with_g' in self.args.note:
#                 self.params[k] = self.params[k] - \
#                     x[k]*(1-self.bias[k]*(1-self.p))
#             elif 'without_g' in self.args.note:
#                 self.params[k] = self.params[k] - \
#                     x[k] + self.bias[k]*(1-self.p)
#             # self.params[k] = self.params[k] - x[k]
#             self.params[k] = self.params[k].mean(dim=0)

#             # self.w_d[k] = self.w_d[k].mean(dim=0)
#             # self.w_lr[k] = self.w_lr[k].mean(dim=0)
#             # self.bias[k] = self.bias[k].mean(dim=0)
#         model = copy.deepcopy(self.last_model)
#         if self.training:
#             load_model(model, self.params)
#             return model
#         else:
#             w = {}
#             for k, v in self.params.items():
#                 self.last_params[k] = v
#                 w[k.replace('-', '.')] = v
#             model.load_state_dict(w)
#             # self.record['delta_w'].append(x)
#             # self.record['u'].append(copy.deepcopy(self.bias))
#         return model

#     def preprocess_gradients(x):
#         p = torch.tensor(10)
#         eps = 1e-6
#         indicator = (x.abs() > torch.exp(-p)).float()
#         x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
#         x2 = x.sign() * indicator + torch.exp(p) * x * (1 - indicator)

#         return torch.cat((x1, x2), 1)
