from simplefl.methods.fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch

class FedDyn(FedAvg):
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
    
    def server_init(self, alpha=0.5):
        self.server.model = self.server.init_model()  
        self.alpha = alpha  # 正则化强度参数
        h = copy.deepcopy(self.server.model)
        self.zero_weights(h)
        self.h = h.state_dict()
        
    def server_update(self):
        u_avg = self.averaging(self.models, w='u')
        w_avg = self.averaging(self.models, w='w')
        m=copy.deepcopy(self.server.model.state_dict())
        model_param={}
        for k,v in self.h.items():
            self.h[k]=v-self.alpha*(u_avg[k]-m[k])
            model_param[k]=w_avg[k]-1/self.alpha*self.h[k]
        self.server.model.load_state_dict(model_param)
            

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        # 初始化存储以前全局模型的副本，用于正则化项
        prox = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # 跳过没有数据的客户端
            return model_l
        
        for E in range(self.E):
            self.fit_feddyn(model_l, train_loader, optimizer, prox, client)
        
        return model_l

    def fit_feddyn(self, model, train_loader, optimizer, prox, client):
        model.train().to(model.device)
        # 只获取状态字典，不将整个模型移到设备，避免显存积累
        prox_params = prox.state_dict()
        
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            output = model(batch)
            label = batch['label']
            loss = model.loss_fn(output, label)
            
            # 计算动态正则化项
            prox_loss = 0.0
            for n, p in model.named_parameters():
                # 只在需要时将参数移到设备，使用完立即释放
                prox_param = prox_params[n].to(p.device)
                prox_loss += self.alpha * 0.5 * torch.norm(p - prox_param) ** 2
                del prox_param  # 显式释放
            
                
            # 计算内积
            grad_loss = 0.0            
            if hasattr(client,'grad'):
                for n, p in model.named_parameters():
                    # 只在需要时将梯度移到设备
                    grad = client.grad[n].to(p.device)
                    grad_loss += torch.sum(p * grad)
                    del grad  # 显式释放
                    
                            
            total_loss = loss + prox_loss+grad_loss
            total_loss.backward()
            optimizer.step()

        # 更新本地梯度近似
        self.update_local_gradient_approximation(client, model, prox)

    def update_local_gradient_approximation(self, client, model, prox):
        # 实现更新本地梯度近似，这里简单地将当前模型参数减去上一轮全局模型参数
        # 并乘以 alpha 系数作为梯度近似
        with torch.no_grad():
            prox_params = prox.state_dict()  # 获取CPU上的参数
            if hasattr(client,'grad'):
                for k, m in model.named_parameters():
                    # 将参数移到CPU进行计算，避免显存积累
                    m_cpu = m.detach().cpu()
                    p_cpu = prox_params[k].cpu()
                    # 确保 client.grad[k] 也在 CPU 上
                    if client.grad[k].device.type != 'cpu':
                        client.grad[k] = client.grad[k].cpu()
                    client.grad[k] = client.grad[k] - self.alpha * (m_cpu - p_cpu)
                    del m_cpu, p_cpu  # 显式释放
            else:
                client.grad={}
                for k, m in model.named_parameters():
                    # 将参数移到CPU进行存储，避免显存积累
                    m_cpu = m.detach().cpu()
                    p_cpu = prox_params[k].cpu()
                    client.grad[k] = -self.alpha * (m_cpu - p_cpu)
                    del m_cpu, p_cpu  # 显式释放
