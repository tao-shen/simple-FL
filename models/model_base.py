import torch
# from torch.tensor import Tensor
# from utils import to_device, plot_roc
# from tqdm import tqdm
# from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import copy
# import random


class MLP(nn.Module):
    def __init__(self, input_size, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, args.hidden_size[0]),
            # nn.BatchNorm1d(args.hidden_size[0])
        )
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_size[0], args.hidden_size[1]),
            # nn.Norm1d(args.hidden_size[1])
        )
        self.fc3 = nn.Linear(args.hidden_size[1], 1)
        self.relu = torch.nn.ReLU()
        #
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        # x = self.dropout(self.relu(self.fc1(input)))
        # x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output


class Patch(nn.Module):
    def __init__(self, patch_size):
        super(Patch, self).__init__()
        self.fc1 = nn.Linear(patch_size, patch_size)
        # self.fc2 = nn.Linear(patch_size, patch_size)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        # x = self.fc2(x)
        output = x + input
        return output

# def embtensor(tensor):
#     if tensor.device.type == 'cuda':
#         return torch.cuda.LongTensor(tensor)
#     else:
#         return torch.LongTensor(tensor)


class Adaptor:
    def __init__(self, patch):
        self.templete = copy.deepcopy(patch)

    def to_tensor(self, x):
        # self.templete = copy.deepcopy(x[0])
        y = []
        # z = torch.tensor([]).to(next(x[0].parameters()).device)
        for p in x:
            z = torch.tensor([]).to(next(x[0].parameters()).device)
            for k, v in p.state_dict().items():
                z = torch.cat((z, torch.flatten(v)), 0)
                # self
            y.append(z)
        y = torch.stack(y, dim=0)
        y.requires_grad = False
        return y

    def to_patch(self, x):
        y = []
        for i in range(x.size()[0]):
            z = self.templete.state_dict()
            start = 0
            for k, v in z.items():
                end = start + v.numel()
                z[k] = x[i, start:end].view_as(z[k])
                start = end
                # self.templete.load_state_dict(z)
            # y.append(copy.deepcopy(self.templete))
            # patch = copy.deepcopy(self.templete)
            # patch.load_state_dict(z)
            y.append(z)
        return y

    # def adaptor(x):
    #     if isinstance(x, Patch):
    #         z = torch.tensor([]).to(next(x.parameters()).device)
    #         for v in x.state_dict().values():
    #             z = torch.cat((z, torch.flatten(v)), 0)
    #         z.requires_grad = False
    #     elif isinstance(x, Tensor):
    #         s = Patch(48)
    #         z = x
    #     return z


class Pooling(nn.Module):
    def __init__(self, pooling_type, dim=1, **kwargs):
        super(Pooling, self).__init__()
        self.dim = dim
        self.pooling_type = pooling_type
        if self.pooling_type == 'mean':
            self.pooling = torch.mean
        if self.pooling_type == 'sum':
            self.pooling = torch.sum
        # if type == 'max':
        #     self.pooling = torch.max
        if self.pooling_type == 'attention':
            if 'patch' in kwargs.keys():
                self.pooling = Attention_Pooling(
                    kwargs['args'], patch=kwargs['patch'])
            else:
                self.pooling = Attention_Pooling(kwargs['args'])

    def forward(self, x, target_item=None):
        if self.pooling_type != 'attention':
            output = self.pooling(x, self.dim)
        else:
            output = self.pooling(x, target_item, self.dim)
        return output


class Attention_Pooling(nn.Module):
    def __init__(self, args, **kwargs):
        super(Attention_Pooling, self).__init__()
        if 'patch' in kwargs.keys():
            self.attention_unit = Attention_Unit(args, patch=kwargs['patch'])
        else:
            self.attention_unit = Attention_Unit(args)

    def forward(self, seq, target_item, dim):
        target_items = target_item.unsqueeze(-2).expand_as(seq)
        # weights = self.attention_unit(target_items.detach(), seq.detach())
        weights = self.attention_unit(target_items, seq)
        weights = torch.softmax(weights, dim=1)
        out = weights*seq
        return out.sum(dim=dim)


class Attention_Unit(nn.Module):

    def __init__(self, args, **kwargs):
        super(Attention_Unit, self).__init__()
        # self.activation = torch.nn.PReLU()
        self.fc1 = nn.Linear(args.item_embed_size*4, args.item_embed_size)
        self.fc2 = nn.Linear(args.item_embed_size, 1)
        self.kwargs = kwargs
        self.activation = torch.nn.ReLU()
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, seq, target_item):
        emb_cat = torch.cat(
            (target_item, seq, target_item-seq, target_item*seq), dim=-1)
        x = self.activation(self.fc1(emb_cat))
        if 'patch' in self.kwargs.keys():
            x = self.kwargs['patch'](x)
        weight = self.fc2(x)
        return weight
