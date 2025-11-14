from simplefl.methods.fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
from models import *


class fedleo(FedAvgM):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()
        if self.args.dataset == 'femnist' or self.args.dataset == 'fashionmnist':
            norm = False
        elif self.args.dataset == 'ml-1m' or self.args.dataset == 'ml-100k':
            norm = True
        self.aggr = PBA(self.server.model, norm=norm)
        self.aggr_optimizer = self.opts['adam'](self.aggr.parameters(),
                                               lr=1e-2,
                                               # momentum=0.5,
                                               # weight_decay=1e-5,
                                               )

    def server_update(self):
        feats = self.to_feats(self.delta_models)
        if 'agg1' in self.args.note:
            with torch.no_grad():
                self.aggr.to(self.args.device).eval()
                feats = to_device(feats, self.args.device)
                self.server.model = self.aggr(feats)
            # if self.round >7:
            #     return
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
        # self.aggr = nn.DataParallel(self.aggr, device_ids=[ 5, 6, 7], output_device=2)
        for E in range(30):
            for idx, batch in enumerate(train_loader):
                self.aggr_optimizer.zero_grad()
                batch = to_device(batch, self.args.device)
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
                # for p1 in model.parameters():
                #     loss2 += torch.norm(p1, p=2)
                # loss += alpha1*loss1+alpha2*loss2
                loss.backward()
                self.aggr_optimizer.step()
            print(E, loss)

    def to_feats(self, models):
        feats = {}
        for k in models[0].state_dict().keys():
            dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
            v = torch.cat(tuple(dicts), dim=0)
            k = k.replace('.', '-')
            feats[k] = v
        return feats
