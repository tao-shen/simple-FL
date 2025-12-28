from .fedavgm import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn


# from models import *
# 定义优化器网络
class OptimizerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OptimizerLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 定义LSTM层
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # 定义全连接层输出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # 传递输入和隐藏状态到LSTM
        hidden = self.lstm(input, hidden)

        # 获取输出
        output = self.fc(hidden[0])

        return output, hidden


# 定义基于坐标的LSTM优化器
class CoordinateLSTMOptimizer(nn.Module):
    def __init__(self, num_params, lstm_dim):
        super().__init__()
        # 对于每个参数坐标,单独设置一个LSTM,但共享权重
        self.lstm = nn.LSTM(1, lstm_dim)
        self.fc = nn.Linear(lstm_dim, 1)  # 输出每个坐标的更新值

        # 对于不同类型的参数(如全连接层和卷积层),设置独立的LSTM和权重
        # ...

    def forward(self, grads):
        # grads是一个tensor,包含所有参数梯度
        outputs = []
        h = None  # 初始化LSTM隐状态
        for grad in torch.unbind(grads):  # 按维解开梯度tensor
            # 对每个坐标,单独输入梯度值到相同的LSTM
            out, h = self.lstm(grad.unsqueeze(0).unsqueeze(2), h)
            out = self.fc(out).squeeze(2)  # 获取更新值
            outputs.append(out)
        return torch.cat(outputs)  # 将所有坐标的更新值连接

    def preprocess_input(gradient):
        # 定义输入预处理函数
        p = 10
        gradient_norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
        gradient_norm_clipped = torch.clamp(gradient_norm, min=1e-10)
        gradient_normalized = gradient / gradient_norm_clipped
        log_gradient_norm = torch.log(gradient_norm_clipped)

        processed_input = torch.cat([log_gradient_norm, gradient_normalized], dim=1)

        return processed_input


class FedLeo(FedAvg):

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()
        self.aggr = Leo(self.server.model, self.args)
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        self.aggr_optimizer = opts[self.args.server_optimizer](
            self.aggr.parameters(),
            lr=self.args.lr_g,
            weight_decay=self.args.weight_decay,
        )

    def server_update(self):
        feats = self.to_feats(self.models)
        if "agg1" in self.args.note:
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


class Leo(nn.Module):
    def __init__(self, init_model, args, norm=False):
        super(Leo, self).__init__()
        self.args = args
        self.p = args.dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.p)
        self.size = {k: v.size() for k, v in init_model.state_dict().items()}
        input_size = sum([v.numel() for v in self.size.values()])
        hidden = int(torch.log2(torch.tensor(input_size)))
        output_size = input_size
        self.weight = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            self.activation,
            nn.Linear(hidden, output_size, bias=False),
        )
        self.templete_model = copy.deepcopy(init_model)

    def forward(self, feats):
        vector = feats2vector(feats)
        bias = self.weight(vector)
        # bias = 0
        w = torch.mean(vector, dim=0)
        w_dict = vector2model_dict(w, self.size)
        model = copy.deepcopy(self.templete_model)
        if self.training:
            load_model(model, w_dict)
            return model
        else:
            model.load_state_dict(w_dict)
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
        d = k.split(".")
        m = model
        for k in d[:-1]:
            m = m._modules[k]
        m._parameters[d[-1]] = v


def feats2vector(feats):
    # Flattening the parameters
    # param_shapes = {k: v.shape[1:] for k, v in feats.items()}
    flattened_params = torch.hstack([p.flatten(1) for p in feats.values()])
    return flattened_params


def vector2model_dict(flattened_params, param_shapes):

    # Reshaping the flattened parameters
    start_idx = 0
    model_dict = {}
    for key, shape in param_shapes.items():
        size = np.prod(shape)
        model_dict[key] = flattened_params[start_idx : start_idx + size].reshape(shape)
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
#         self.templete_model = copy.deepcopy(init_model)
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
#         for key, v in self.templete_model.state_dict().items():
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
#             self.register_parameter('templete_model_'+k, self.last_params[k])

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
#         model = copy.deepcopy(self.templete_model)
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
