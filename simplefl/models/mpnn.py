from .model_base import *


def message_func(edges):
    out = dict(edges.src)
    return out


def reduce_func(nodes):
    out = {}
    for k, v in nodes.mailbox.items():
        out[k] = torch.mean(v, dim=1)
    return out


class GraphAvg_layer(nn.Module):
    def __init__(self):
        super(GraphAvg_layer, self).__init__()

    def forward(self, g, h):
        g = copy.deepcopy(g)
        for k, v in h.items():
            g.ndata[k] = v
        g.update_all(message_func, reduce_func)
        h = dict(g.ndata)
        return h

class PGA_layer(nn.Module):
    def __init__(self, templete,):
        super(PGA_layer, self).__init__()
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.weight = nn.ModuleDict()
        self.size = {}
        self.templete = [copy.deepcopy(templete)]

        for k, v in templete.state_dict().items():
            size = v.size().numel()
            k = k.replace('.', '-')
            self.size[k] = v.size()
            self.weight[k] = nn.Sequential(
                nn.Linear(size, int(torch.log2(torch.tensor(size))+1)),
                # self.dropout,
                nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                # nn.BatchNorm1d(32, track_running_stats=False),
                self.activation,
                nn.Linear(int(torch.log2(torch.tensor(size))+1),
                          int(torch.log2(torch.tensor(size))+1)),
                # self.dropout,
                nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                # nn.BatchNorm1d(32, track_running_stats=False),
                self.activation,
                nn.Linear(int(torch.log2(torch.tensor(size))+1),
                          v.size().numel()),
                # self.dropout,
                # nn.LayerNorm(v.size().numel()),
                # nn.BatchNorm1d(v.size().numel(), track_running_stats=False),
                # self.activation,
            )

    def forward(self, g, x): 
        h, dx = {}, {}
        g = copy.deepcopy(g)
        for k, v in self.weight.items():
            dx[k] = v(x[k].flatten(1)).view_as(x[k])
            h[k] = self.activation(dx[k])
            g.ndata[k] = h[k]

        g.update_all(message_func, reduce_func)
        h = dict(g.ndata)
        return h


class PBA(nn.Module):
    def __init__(self, templete, norm=False):
        super(PBA, self).__init__()
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.weight = nn.ModuleDict()
        self.size = {}
        self.templete = [copy.deepcopy(templete)]
        self.dx={}

        for k, v in templete.state_dict().items():
            size = v.size().numel()
            k = k.replace('.', '-')
            self.size[k] = v.size()
            if norm:
                self.weight[k] = nn.Sequential(
                    nn.Linear(size, int(torch.log2(torch.tensor(size))+1), bias=False),
                    # self.dropout,
                    nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                    # nn.BatchNorm1d(32, track_running_stats=False),
                    self.activation,
                    nn.Linear(int(torch.log2(torch.tensor(size))+1),
                            int(torch.log2(torch.tensor(size))+1), bias=False),
                    # self.dropout,
                    nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                    # nn.BatchNorm1d(32, track_running_stats=False),
                    self.activation,
                    nn.Linear(int(torch.log2(torch.tensor(size))+1),
                            v.size().numel(), bias=False),
                    # self.dropout,
                    # nn.LayerNorm(v.size().numel()),
                    # nn.BatchNorm1d(v.size().numel(), track_running_stats=False),
                    # self.activation,  
                )              
            else:    
                self.weight[k] = nn.Sequential(
                    nn.Linear(size, int(torch.log2(torch.tensor(size))+1), bias=False),
                    # self.dropout,
                    # nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                    # nn.BatchNorm1d(int(torch.log2(torch.tensor(size))+1), track_running_stats=False),
                    self.activation,
                    nn.Linear(int(torch.log2(torch.tensor(size))+1),
                            int(torch.log2(torch.tensor(size))+1), bias=False),
                    # self.dropout,
                    # nn.LayerNorm(int(torch.log2(torch.tensor(size))+1)),
                    # nn.BatchNorm1d(int(torch.log2(torch.tensor(size))+1), track_running_stats=False),
                    self.activation,
                    nn.Linear(int(torch.log2(torch.tensor(size))+1),
                            v.size().numel(), bias=False),
                    # self.dropout,
                    # nn.LayerNorm(v.size().numel()),
                    # nn.BatchNorm1d(v.size().numel(), track_running_stats=False),
                    # self.activation,
                )

    def forward(self, x):
        h = {}
        for k, v in self.weight.items():
            p=self.preprocess_gradients(x[k])
            self.dx[k] = v(x[k].flatten(1)).view_as(x[k])
            h[k] = self.dx[k] + x[k]
            # h[k] = dx[k]
            h[k] = h[k].mean(dim=0)
        model = copy.deepcopy(self.templete[0])
        if self.training:
            load_model(model, h)
            return model
        else:
            w={}
            for k,v in h.items():
                w[k.replace('-','.')]=h[k]
            model.load_state_dict(w)
            return model

    def preprocess_gradients(self, x):
        p = torch.tensor(10)
        eps = 1e-6
        indicator = (x.abs() > torch.exp(-p)).float()
        x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
        x2 = x.sign() * indicator + torch.exp(p) * x * (1 - indicator)

        return torch.cat((x1, x2), 1)

class PGA(nn.Module):
    def __init__(self, templete, n_layers,):
        super(PGA, self).__init__()
        self.activation = torch.relu
        self.templete = [copy.deepcopy(templete)]
        self.layers = nn.ModuleDict()
        self.layers['input_layer'] = PGA_layer(templete,)
        for i in range(n_layers):
            self.layers['hidden_layer_{}'.format(
                i)] = PGA_layer(templete,)
        self.layers['output_layer'] = PGA_layer(templete,)

    def forward(self, g, x):
        h = copy.deepcopy(x)
        for k,layer in self.layers.items():
            h = layer(g, h)
        for k, v in h.items():
            h[k] = v+x[k]
            h[k] = h[k].mean(dim=0)
        model = copy.deepcopy(self.templete[0])
        if self.training:
            load_model(model, h)
            return model
        else:
            w = {}
            for k, v in h.items():
                w[k.replace('-', '.')] = h[k]
            model.load_state_dict(w)
            return model


class GraphAvg(nn.Module):
    def __init__(self, n_layers):
        super(GraphAvg, self).__init__()
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers['layer_{}'.format(i)] = GraphAvg_layer()

    def forward(self, g, x):
        h = x
        for k, layer in self.layers.items():
            h = layer(g, h)
        return h


def load_model(model, model_dict):
    for k, v in model_dict.items():
        d = k.split('-')
        m = model
        for k in d[:-1]:
            m = m._modules[k]
        m._parameters[d[-1]] = v
        # m._parameters[d[-1]] = v.clone()
        # m._parameters[d[-1]].retain_grad()
