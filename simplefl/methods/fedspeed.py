from simplefl.methods.fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedSpeed(FedAvg):
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
    
    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        if not hasattr(client,'grad_prox'):
            client.grad_prox = {name: torch.zeros_like(param).cpu() for name, param in model_l.named_parameters()}  # 初始化 ĝt−1_i

        for E in range(self.E):
            model_l = self.fit_fedspeed(model_l, train_loader, optimizer, model_g, client)
        return model_l

    def fit_fedspeed(self, model, train_loader, optimizer, model_g, client ,alpha=0.9,rho=0.1,lambda_=100):
        
        model.train().to(model.device)
        model_g = model_g.to(model.device).state_dict()
        
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc='local_update')
        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            
            # 计算梯度1
            output = model(batch)
            label = batch['label']
            loss = model.loss_fn(output, label)
            loss.backward()
            grad_1 = {name: param.grad.data for name, param in model.named_parameters()}
            grad_extra = {name: param.data + rho * param.grad.data for name, param in model.named_parameters()}
            optimizer.zero_grad()
            
            # 计算梯度2
            model_extra=copy.deepcopy(model)
            model_extra.load_state_dict(grad_extra)
            output_extra = model_extra(batch)
            loss_extra = model_extra.loss_fn(output_extra, label)
            loss_extra.backward()
            grad_2 = {name: param.grad.data for name, param in model_extra.named_parameters()}
            optimizer.zero_grad()
            
            # 计算拟梯度
            for n, p in model.named_parameters():
                grad_combined = (1 - alpha) * grad_1[n] + alpha * grad_2[n]
                p.grad.data = grad_combined - client.grad_prox[n].to(self.args.device) + 1 / lambda_ * (p.data - model_g[n]) 
            
            optimizer.step()
        
        # 更新grad_prox和参数项
        for n, p in model.named_parameters():
            delta=p.data - model_g[n]
            client.grad_prox[n] -= 1 / lambda_ * delta.cpu()
            p.data-=lambda_*client.grad_prox[n].to(self.args.device)
        return model
