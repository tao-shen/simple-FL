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

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
        if args.FL_validate_clients:
            self.is_train = False

        else:
            self.is_train = True
        # self.is_train=False

    def server_init(self):
        # 初始化模型
        self.server.model = self.server.init_model()
        # 初始化代理数据
        self.server.proxy_data = self.server.init_proxy_data()
        input_size, hidden_size, output_size, num_layers=2, 50, 1, 2 
        # 初始化优化器
        # num_layers包括输入层和隐层，但还需要接入输出层
        self.agg_opt = CoordinateLSTMOptimizer(
            input_size, hidden_size, output_size, num_layers, self.args
        ).to(self.args.device)
        if self.is_train == False:
            self.agg_opt = torch.load(
                "./assets/"
                + "_".join(
                    ["aggr", self.args.dataset, self.args.iid, str(self.args.seed)]
                )
                + ".pt",
                map_location="cpu",
            ).to(self.args.device)
        self.opts = {"sgd": torch.optim.SGD, "adam": torch.optim.AdamW}
        self.agg_opt_optimizer = self.opts[self.args.server_optimizer](
            self.agg_opt.parameters(),
            lr=0.001,
            # weight_decay=self.args.weight_decay,
        )
        # 初始化代理损失
        self.proxy_loss = 0
        # 初始化时间步长
        self.t = 0
        # 截断时间步长
        self.trunced_t=3

        # 初始化优化状态
        # v_m=self.server.model.to_vector().data.to(self.args.device)
        # h0=torch.zeros(len(v_m),hidden_size, dtype=torch.float).to(self.args.device)
        # h1=torch.zeros(len(v_m),hidden_size, dtype=torch.float).to(self.args.device)
        # h1=v_m.unsqueeze(dim=-1).expand_as(h0)
        # self.hx = [h0,h1]*num_layers
        self.hx=[None]*num_layers
        model_param=self.server.model.to_vector().data.to(self.args.device)
        f=torch.zeros_like(model_param)
        i=torch.zeros_like(model_param)
        self.hx.append(torch.stack([f,i, model_param], dim=0))
        self.trunced_hx=copy.deepcopy(self.hx)
        self.model_checkpoints=[]
        self.grads_checkpoints=[]

    def server_update(self):

        ### TODO lr和w_decay需要被正确显示，需要考虑is_train的状态，加入模型异质性的学习（Attention），拓展到不同架构的模型。
        # 先加1防止后面需要加入self.t!=0的判断
        self.t += 1 

        # 得到模型参数和梯度
        model_param = self.server.model.to_vector().data.to(self.args.device)
        local_grads = [delta_model.to_vector() for delta_model in self.delta_models]
        local_grads = torch.stack(local_grads).data.T.to(self.args.device)

        # 训练聚合器
        if self.is_train:
            self.model_checkpoints.append(model_param.cpu())
            self.grads_checkpoints.append(local_grads.cpu())
            if self.t % self.trunced_t == 0:
                self.aggr_training()
                self.model_checkpoints=[]
                self.grads_checkpoints=[]

        # 聚合模型梯度
        with torch.set_grad_enabled(self.is_train):
            # hx[-1]包含了lr, weight_decay, model_param, hx[:-1]是前两层lstm的h和c
            vec, hx = self.agg_opt(local_grads, self.hx)
            self.hx=[h.data for h in hx]
            self.server.model.from_vector(vec.data.cpu())  # 这句执行后，server.model里的参数还是leaf

        # 记录学到的参数
        try:
            learned_lr_g=torch.mean(self.hx[-1][1].data**2).item()
            learned_w_decay=torch.mean(self.hx[-1][0].data**2).item()
            self.recorder({'learned_lr_g':learned_lr_g,
                           'learned_w_decay':learned_w_decay})
        except:
            pass

    def aggr_training(self):
        proxy_model = copy.deepcopy(self.server.model)
        loss_fun = proxy_model.loss_fn
        proxy_set = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            proxy_set, batch_size=self.args.server_batch_size, shuffle=True
        )
        for E in range(5):
            for batch in train_loader:
                loss = 0         
                batch = to_device(batch, self.args.device)
                hx = copy.deepcopy(self.trunced_hx)#这里必须要deepcopy因为在aggr中hx被重新赋值了
                for local_grads in self.grads_checkpoints:
                    local_grads=local_grads.to(self.args.device)
                    model_param=hx[-1][-1]
                    vec, hx = self.agg_opt(local_grads, hx)
                    self.proxy_model_from_vector(proxy_model, vec)
                    pred = proxy_model(batch)
                    label = batch["label"]
                    pred_loss = loss_fun(pred, label)
                    prox_loss=0
                    # prox_loss=torch.sum((model_param-vec)**2)
                    loss += pred_loss + prox_loss                    
                loss/=len(self.model_checkpoints)
                loss.backward()
                self.agg_opt_optimizer.step()
                self.agg_opt_optimizer.zero_grad()
                print(E, loss.data)
        self.server.model.from_vector(vec.cpu())
        self.hx = [h.data for h in hx]
        self.trunced_hx = [h.data for h in hx]

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

    def proxy_model_from_vector(self, model, vec):
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

    def load_aggregator(self, path):
        self.agg_opt = torch.load(path)
        self.is_train = False

    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False

    # def aggr_training(self, vec):
    #     proxy_model = copy.deepcopy(self.server.model)
    #     self.proxy_model_from_vector(proxy_model, vec)  # 这句执行后，模型参数不是leaf
    #     self.proxy_loss += self.eval_proxy_loss(proxy_model, self.server.proxy_data)

    #     if self.t != 0 and self.t % self.trunced_t == 0:
    #         try:
    #             self.agg_opt_optimizer.zero_grad()
    #             self.proxy_loss/=self.trunced_t
    #             self.proxy_loss.backward()
    #             self.agg_opt_optimizer.step()
    #             self.proxy_loss = 0
    #             self.hx[0] = self.hx[0].data
    #             self.hx[1] = self.hx[1].data
    #         except:
    #             pass
    #     self.t += 1
