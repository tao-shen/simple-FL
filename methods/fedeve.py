from .fedavgm import FedAvgM
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedEve(FedAvgM):
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def method_init(self):
        self.args.lr_g = 1
        self.args.F = 1
        self.args.H = 1
        self.args.Q = 1
        self.args.R = 1

    def server_init(self):
        self.k = {}
        self.server.model = self.server.init_model()
        m = copy.deepcopy(self.server.model)
        self.zero_weights(m)
        self.m = m.state_dict()
        # sigma = copy.deepcopy(self.server.model)
        # self.zero_weights(sigma)
        # self.sigma = sigma.state_dict()
        self.sigma = 0
        # m = copy.deepcopy(self.server.model)
        # self.zero_weights(m)
        # self.m = m.state_dict()
        # self.m = copy.deepcopy(self.m)
        self.a = copy.deepcopy(self.m)
        self.s = copy.deepcopy(self.m)
        # self.round = 1
        self.args.recorder["kal_Q"] = []
        self.args.recorder["kal_R"] = []
        self.args.recorder["kal_sigma"] = []
        self.args.recorder["kal_K"] = []
        # self.sigma = copy.deepcopy(self.m)

        # # model-wise
        # params = [p.flatten() for p in self.server.model.parameters()]
        # self.sigma = torch.var(torch.hstack(params))

        # layer-wise
        # sigma = copy.deepcopy(self.server.model.state_dict())
        # for k in sigma.keys():
        #     sigma[k] = torch.var(sigma[k])
        #     if torch.isnan(sigma[k]):
        #         sigma[k] = 0
        #     self.k[k] = 1
        # self.sigma = sigma

    def clients_update(self):
        lr_g = self.args.lr_g
        self.delta_models = []
        w = self.server.model.state_dict()
        self.w_predict = {}
        for k in w:
            self.w_predict[k] = w[k] - lr_g * self.m[k]
        predict = copy.deepcopy(self.server.model)
        predict.load_state_dict(self.w_predict)
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], predict)
            delta_model = self.diff_model(model, predict)
            self.delta_models.append(delta_model)

    def server_update(self):
        w_t = self.server.model.state_dict()
        delta_w_t = self.averaging(self.delta_models)

        self.R = self.r()
        self.Q = self.q(delta_w_t)
        lr_g = self.args.lr_g

        # model-wise
        w = {}
        # self.k = {}
        # for key in self.sigma.keys():
        #     self.sigma[key] = self.sigma[key] + self.Q
        #     self.k[key] = self.sigma[key]/(self.sigma[key]+self.R)
        #     self.sigma[key] = (1-self.k[key])*self.sigma[key]
        # for key in w_t.keys():
        #     # self.w_predict[key] = self.w_updated[key]+
        #     # self.k[key] = 0.9
        #     self.m[key] = self.m[key]+self.k[key] * \
        #         (delta_w_t[key]-self.m[key])
        #     w[key] = w_t[key]-lr_g*self.m[key]
        self.sigma = self.sigma + self.Q
        self.k = self.sigma / (self.sigma + self.R)
        self.sigma = (1 - self.k) * self.sigma

        self.args.recorder["kal_Q"] += [self.Q.item()]
        self.args.recorder["kal_sigma"] += [self.sigma.item()]
        self.args.recorder["kal_R"] += [self.R.item()]
        self.args.recorder["kal_K"] += [self.k.item()]
        for key in w_t.keys():
            # self.w_predict[key] = self.w_updated[key]+
            # self.k[key] = 0.9
            self.m[key] = self.m[key] + self.k * (delta_w_t[key] - self.m[key])
            w[key] = w_t[key] - lr_g * self.m[key]
        self.server.model.load_state_dict(w)

        # # layer-wise
        # w = {}
        # # self.k = {}
        # for key in w_t.keys():
        #     # self.w_predict[key] = self.w_updated[key]+
        #     self.sigma[key] = self.sigma[key] + self.Q[key]
        #     self.k[key] = self.sigma[key]/(self.sigma[key]+self.R[key])
        #     # self.k[key] = 0.9
        #     self.m[key] = self.m[key]+self.k[key] * \
        #         (delta_w_t[key]-self.m[key])
        #     w[key] = w_t[key]-lr_g*self.m[key]
        #     self.sigma[key] = (1-self.k[key])*self.sigma[key]
        # self.server.model.load_state_dict(w)

    # def r(self):
    #     r = {}
    #     with torch.no_grad():
    #         model = copy.deepcopy(self.delta_models[0])
    #         w = model.state_dict()
    #         num_sum = sum([m.train_num for m in self.delta_models])
    #         for key in w.keys():
    #             w[key] = 0
    #             for m in self.delta_models:
    #                 w[key] += m.state_dict()[key]
    #                 # w[key] += m.state_dict()[key]*m.train_num
    #             w[key] /= len(self.delta_models)
    #             # w[key] /= num_sum

    #             # # model-wise
    #             # r = 0

    #             # p1 = torch.hstack([p.flatten() for p in w.values()])
    #             # for m in self.delta_models:
    #             #     p2 = torch.hstack([p.flatten() for p in m.parameters()])
    #             #     r += (p1-p2)**2
    #             # r/= len(self.delta_models)**2

    #             # layer-wise
    #             r[key] = 0
    #             for m in self.delta_models:
    #                 # r[key] += torch.mean((w[key]-m.state_dict()[key])**2)
    #                 r[key] += (w[key]-m.state_dict()[key])**2
    #                 # r[key] += torch.mean((w[key]-m.state_dict()
    #                 #                      [key])**2*m.train_num**2)
    #             r[key] /= len(self.delta_models)**2
    #             # r[key] /= num_sum**2
    #     return r

    def r(self):
        r = {}
        with torch.no_grad():
            model = copy.deepcopy(self.delta_models[0])
            w = model.state_dict()
            num_sum = sum([m.train_num for m in self.delta_models])
            for key in w.keys():
                w[key] = 0
                for m in self.delta_models:
                    w[key] += m.state_dict()[key]
                    # w[key] += m.state_dict()[key]*m.train_num
                w[key] /= len(self.delta_models)
                # w[key] /= num_sum

                # # layer-wise
                # r[key] = 0
                # for m in self.delta_models:
                #     # r[key] += torch.mean((w[key]-m.state_dict()[key])**2)
                #     r[key] += (w[key]-m.state_dict()[key])**2
                #     # r[key] += torch.mean((w[key]-m.state_dict()
                #     #                      [key])**2*m.train_num**2)
                # r[key] /= len(self.delta_models)**2
                # # r[key] /= num_sum**2

            # model-wise
            r = 0
            p1 = torch.hstack([p.flatten() for p in w.values()])
            for m in self.delta_models:
                p2 = torch.hstack([p.flatten() for p in m.parameters()])
                r += (p1 - p2) ** 2
            r /= len(self.delta_models) ** 2
            r = torch.mean(r)

        return r

    def q(self, delta_w_t):
        q = {}
        # model-wise
        p1 = torch.hstack([p.flatten() for p in self.m.values()])
        p2 = torch.hstack([p.flatten() for p in delta_w_t.values()])
        q = torch.mean((p2) ** 2 / len(self.delta_models))

        # # layer-wise
        # for k in delta_w_t.keys():
        #     # q[k] = torch.mean(((self.m[k]-delta_w_t[k]))
        #     #                   ** 2/len(self.delta_models))
        #     # q[k] = (self.m[k]-delta_w_t[k]) ** 2/len(self.delta_models)
        #     q[k] = (delta_w_t[k]) ** 2/len(self.delta_models)
        #     # q[k] = torch.mean(((self.m[k]-delta_w_t[k]))**2)

        return q

    # def q(self, a):
    #     a_bar = copy.deepcopy(self.a)
    #     s_bar = copy.deepcopy(self.s)
    #     t = self.round
    #     for k in a.keys():
    #         a_bar[k] = ((t-1) * self.a[k]+a[k])/t
    #         s_bar[k] = ((t-1) * (self.s[k]+self.a[k]**2) +
    #                     a[k]**2)/t-a_bar[k]**2
    #     self.round += 1
    #     self.a = a_bar
    #     self.s = s_bar
    #     return self.s

    # def local_update(self, client, model_g):
    #     model_l = copy.deepcopy(model_g)
    #     prox = copy.deepcopy(model_g)
    #     opts = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}
    #     optimizer = opts[self.args.local_optimizer](
    #         model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
    #     # optimizer = torch.optim.Adam(
    #     #     model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
    #     try:
    #         train_loader = DataLoader(
    #             Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
    #     except:  # skip dataless client
    #         return model_l
    #     for E in range(self.E):
    #         self.fit_prox(model_l, train_loader, optimizer, prox)
    #     return model_l

    # def fit_prox(self, model, train_loader, optimizer, prox):
    #     mu = self.args.mu
    #     model.train().to(model.device)
    #     prox = prox.to(model.device).state_dict()
    #     description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
    #     loss_total, avg_loss = 0.0, 0.0
    #     epochs = tqdm(train_loader, leave=False, desc='local_update')
    #     for idx, batch in enumerate(epochs):
    #         optimizer.zero_grad()
    #         batch = to_device(batch, model.device)
    #         output = model(batch)
    #         label = batch['label']
    #         loss = model.loss_fn(output, label)
    #         prox_loss = 0.0
    #         for n, p in model.named_parameters():
    #             prox_loss += torch.square(prox[n]-p).sum()
    #         loss += mu/2*prox_loss
    #         loss.backward()
    #         optimizer.step()
    #         loss_total += loss.item()
    #         loss_avg = loss_total / (idx + 1)
    #     model.train_num = len(train_loader.dataset)
