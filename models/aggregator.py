from .model_base import *


class PA(nn.Module):
    def __init__(self, init_model, args, norm=False):
        super(PA, self).__init__()
        self.args = args
        self.p = args.dropout
        self.activation = nn.Tanh()
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
            self.record['delta_w'].append(x)
            self.record['u'].append(copy.deepcopy(self.bias))
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


# class PA(nn.Module):
#     def __init__(self, init_model, norm=False):
#         super(PA, self).__init__()
#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.weight = nn.ModuleDict()
#         self.size = {}
#         self.last_model = copy.deepcopy(init_model)
#         self.meta_params = {}
#         self.meta_w_d = {}
#         self.meta_w_lr = {}
#         self.meta_bias = {}
#         self.params = {}
#         self.w_d = {}
#         self.w_lr = {}
#         self.bias = {}
#         self.identity = nn.Identity()
#         for key, v in self.last_model.state_dict().items():
#             # size = v.size().numel()
#             k = key.replace('.', '-')
#             # self.size[k] = v.size()
#             self.weight[k] = nn.Sequential(
#                 nn.Linear(5, 3),
#                 self.activation,
#                 nn.Linear(3, 3),)
#             self.meta_params[k] = nn.Parameter(copy.deepcopy(v))
#             self.meta_w_d[k] = nn.Parameter(torch.randn_like(v))
#             self.meta_w_lr[k] = nn.Parameter(torch.randn_like(v))
#             self.meta_bias[k] = nn.Parameter(torch.randn_like(v))
#             self.register_parameter('last_model_'+k, self.meta_params[k])
#             self.register_parameter('meta_w_d_'+k, self.meta_w_d[k])
#             self.register_parameter('meta_w_lr_'+k, self.meta_w_lr[k])
#             self.register_parameter('meta_bias_'+k, self.meta_bias[k])
#             self.params[k] = self.meta_params[k]
#             self.w_d[k] = self.meta_w_d[k]
#             self.w_lr[k] = self.meta_w_lr[k]
#             self.bias[k] = self.meta_bias[k]

#     def forward(self, x):
#         h = {}
#         for k, v in self.weight.items():
#             self.params[k] = self.params[k].expand_as(x[k])
#             self.w_d[k] = self.w_d[k].expand_as(x[k])
#             self.w_lr[k] = self.w_lr[k].expand_as(x[k])
#             self.bias[k] = self.bias[k].expand_as(x[k])
#             feats = torch.stack(
#                 (self.params[k], x[k], self.w_d[k], self.w_lr[k], self.bias[k]), dim=-1)
#             self.w_d[k], self.w_lr[k], self.bias[k] = torch.unbind(
#                 self.weight[k](feats), -1)
#             self.params[k] = self.w_d[k]*self.params[k] - \
#                 self.w_lr[k]*x[k] + self.bias[k]
#             self.params[k] = self.params[k].mean(dim=0)
#             self.w_d[k] = self.w_d[k].mean(dim=0)
#             self.w_lr[k] = self.w_lr[k].mean(dim=0)
#             self.bias[k] = self.bias[k].mean(dim=0)
#         model = copy.deepcopy(self.last_model)
#         if self.training:
#             load_model(model, self.params)
#             return model
#         else:
#             w = {}
#             for k, v in h.items():
#                 w[k.replace('-', '.')] = self.params[k]
#             model.load_state_dict(w)
#         return model
