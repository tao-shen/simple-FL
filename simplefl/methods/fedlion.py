from .fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, List


# ==================== 常量定义 ====================

# FedLion 默认参数
DEFAULT_BETA1 = 0.9  # 动量系数 β1
DEFAULT_BETA2 = 0.99  # 动量系数 β2


# ==================== FedLion 主类定义 ====================

class FedLion(FedAvg):
    """
    FedLion: Federated Lion Optimizer（联邦 Lion 优化器）
    
    根据算法伪代码实现的FedLion方法：
    
    Algorithm 1 FedLion:
    ===================
    1. 初始化全局模型 x0 和全局动量 m0 = 0
    
    2. 对于每一轮通信 t = 1 to T:
       a. 客户端侧（对于每个被选中的客户端 i）:
          - 接收全局模型 x_{t-1} 和全局动量 m_{t-1}
          - 初始化本地模型 x_{t-1,0}^i = x_{t-1}
          - 初始化本地动量 m_{t-1,0}^i = m_{t-1}
          - 对于每个本地步 s = 1 to E:
            * 计算梯度 g_{t-1,s}^i (使用batch size B)
            * h_{t-1,s}^i = β1 * m_{t-1,s-1}^i + (1 - β1) * g_{t-1,s}^i
            * h_{t-1,s}^i = Sign(h_{t-1,s}^i)
            * x_{t-1,s}^i = x_{t-1,s-1}^i - γ * h_{t-1,s}^i
            * m_{t-1,s}^i = β2 * m_{t-1,s-1}^i + (1 - β2) * g_{t-1,s}^i
          - 计算更新差异: Δ_{t-1}^i = (x_{t-1} - x_{t-1,E}^i) / γ (整数化)
          - 发送 Δ_{t-1}^i 和 m_{t-1,E}^i 到服务器
       
       b. 服务器侧:
          - x_t = x_{t-1} - (γ / n) * Σ_{i=1}^n Δ_{t-1}^i
          - m_t = (1 / n) * Σ_{i=1}^n m_{t-1,E}^i
    
    Attributes:
        beta1: 动量系数 β1 (默认 0.9)
        beta2: 动量系数 β2 (默认 0.99)
        global_momentum: 全局动量状态字典
    """
    
    def __init__(self, server, clients, args):
        """
        初始化 FedLion
        
        Args:
            server: 服务器对象
            clients: 客户端列表
            args: 配置参数对象
        """
        super().__init__(server, clients, args)
        
        # 从配置中获取 FedLion 参数，如果没有则使用默认值
        self.beta1 = getattr(args, 'lion_beta1', DEFAULT_BETA1)
        self.beta2 = getattr(args, 'lion_beta2', DEFAULT_BETA2)
        
        # 全局动量（初始化为None，在server_init中初始化）
        self.global_momentum = None
        
        # 打印 FedLion 配置信息
        print(f"FedLion 初始化完成:")
        print(f"  - 动量系数 (beta1): {self.beta1}")
        print(f"  - 动量系数 (beta2): {self.beta2}")
        print(f"  - 学习率 (gamma): {self.args.lr_l}")

    def server_init(self):
        """
        服务器初始化：初始化全局模型和全局动量
        
        根据算法伪代码：
        - 初始化全局模型 x0 (随机初始化)
        - 初始化全局动量 m0 = 0
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()
        
        # 初始化全局动量为零（与模型参数形状相同）
        self.global_momentum = {}
        with torch.no_grad():
            for name, param in self.server.model.named_parameters():
                self.global_momentum[name] = torch.zeros_like(param.data)

    def local_update(self, client, model_g):
        """
        客户端本地更新：根据算法伪代码实现
        
        算法流程：
        =========
        1. 接收全局模型 x_{t-1} 和全局动量 m_{t-1}
        2. 初始化本地模型 x_{t-1,0}^i = x_{t-1}
        3. 初始化本地动量 m_{t-1,0}^i = m_{t-1}
        4. 对于每个本地步 s = 1 to E:
           - 计算梯度 g_{t-1,s}^i
           - h_{t-1,s}^i = β1 * m_{t-1,s-1}^i + (1 - β1) * g_{t-1,s}^i
           - h_{t-1,s}^i = Sign(h_{t-1,s}^i)
           - x_{t-1,s}^i = x_{t-1,s-1}^i - γ * h_{t-1,s}^i
           - m_{t-1,s}^i = β2 * m_{t-1,s-1}^i + (1 - β2) * g_{t-1,s}^i
        5. 计算更新差异: Δ_{t-1}^i = (x_{t-1} - x_{t-1,E}^i) / γ
        6. 返回更新差异和最终动量
        
        Args:
            client: 客户端对象，包含本地训练数据
            model_g: 全局模型
        
        Returns:
            delta: 更新差异 Δ_{t-1}^i (字典格式，按参数名组织)
            momentum: 最终本地动量 m_{t-1,E}^i (字典格式，按参数名组织)
        """
        # 复制全局模型到本地
        model_l = copy.deepcopy(model_g)
        
        # 准备数据加载器
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), 
                batch_size=self.args.local_batch_size, 
                shuffle=True
            )
        except:  # 跳过没有数据的客户端
            # 返回零更新和零动量
            delta = {}
            momentum = {}
            for name, param in model_g.named_parameters():
                delta[name] = torch.zeros_like(param.data)
                momentum[name] = torch.zeros_like(param.data)
            return delta, momentum
        
        # 将模型移动到正确的设备上
        model_l = model_l.to(self.args.device)
        
        # 初始化本地动量（从全局动量复制）
        local_momentum = {}
        for name, param in model_l.named_parameters():
            if self.global_momentum is not None and name in self.global_momentum:
                local_momentum[name] = self.global_momentum[name].clone().to(self.args.device)
            else:
                local_momentum[name] = torch.zeros_like(param.data)
        
        # 保存初始模型参数（用于计算更新差异）
        initial_params = {}
        for name, param in model_l.named_parameters():
            initial_params[name] = param.data.clone()
        
        # 学习率
        gamma = self.args.lr_l
        
        # 执行本地训练 E 步
        for epoch in range(self.E):
            model_l.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # 将 batch 移到设备上
                batch = to_device(batch, self.args.device)
                
                # 清零梯度
                model_l.zero_grad()
                
                # 前向传播：计算预测值
                output = model_l(batch)
                labels = batch["label"]
                
                # 计算损失
                loss = model_l.loss_fn(output, labels)
                
                # 反向传播：计算梯度
                loss.backward()
                
                # 根据算法伪代码更新参数和动量
                with torch.no_grad():
                    for name, param in model_l.named_parameters():
                        if param.grad is None:
                            continue
                        
                        # 获取梯度
                        grad = param.grad.data
                        
                        # 步骤 10: h_{t-1,s}^i = β1 * m_{t-1,s-1}^i + (1 - β1) * g_{t-1,s}^i
                        h = self.beta1 * local_momentum[name] + (1 - self.beta1) * grad
                        
                        # 步骤 11: h_{t-1,s}^i = Sign(h_{t-1,s}^i)
                        h = torch.sign(h)
                        
                        # 步骤 12: x_{t-1,s}^i = x_{t-1,s-1}^i - γ * h_{t-1,s}^i
                        param.data = param.data - gamma * h
                        
                        # 步骤 13: m_{t-1,s}^i = β2 * m_{t-1,s-1}^i + (1 - β2) * g_{t-1,s}^i
                        local_momentum[name] = self.beta2 * local_momentum[name] + (1 - self.beta2) * grad
        
        # 计算更新差异: Δ_{t-1}^i = (x_{t-1} - x_{t-1,E}^i) / γ
        delta = {}
        final_momentum = {}
        for name, param in model_l.named_parameters():
            # 计算更新差异（整数化，这里我们保持浮点数，实际应用中可能需要量化）
            # 确保所有张量在同一设备上计算
            initial_param_cpu = initial_params[name].cpu()
            param_cpu = param.data.cpu()
            delta[name] = (initial_param_cpu - param_cpu) / gamma
            final_momentum[name] = local_momentum[name].cpu()
        
        return delta, final_momentum

    def clients_update(self):
        """
        客户端更新：收集所有候选客户端的更新差异和动量
        
        遍历所有候选客户端，执行本地训练并收集更新差异和最终动量。
        """
        self.deltas = []  # 存储所有客户端的更新差异
        self.momentums = []  # 存储所有客户端的最终动量
        
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # 执行本地更新，返回更新差异和最终动量
            delta, momentum = self.local_update(self.clients[k], self.server.model)
            self.deltas.append(delta)
            self.momentums.append(momentum)

    def server_update(self):
        """
        服务器更新：根据算法伪代码聚合更新并更新全局模型和动量
        
        算法流程：
        =========
        步骤 19: x_t = x_{t-1} - (γ / n) * Σ_{i=1}^n Δ_{t-1}^i
        步骤 20: m_t = (1 / n) * Σ_{i=1}^n m_{t-1,E}^i
        
        其中：
        - n: 参与本轮训练的客户端数量
        - γ: 学习率
        - Δ_{t-1}^i: 客户端 i 的更新差异
        - m_{t-1,E}^i: 客户端 i 的最终动量
        """
        n = len(self.deltas)  # 参与训练的客户端数量
        if n == 0:
            return
        
        gamma = self.args.lr_l
        
        # 聚合更新差异
        aggregated_delta = {}
        for name in self.deltas[0].keys():
            # 计算所有客户端的更新差异的平均值
            delta_sum = torch.zeros_like(self.deltas[0][name])
            for delta in self.deltas:
                delta_sum += delta[name]
            aggregated_delta[name] = delta_sum / n
        
        # 聚合动量
        aggregated_momentum = {}
        for name in self.momentums[0].keys():
            # 计算所有客户端的动量的平均值
            momentum_sum = torch.zeros_like(self.momentums[0][name])
            for momentum in self.momentums:
                momentum_sum += momentum[name]
            aggregated_momentum[name] = momentum_sum / n
        
        # 更新全局模型: x_t = x_{t-1} - (γ / n) * Σ_{i=1}^n Δ_{t-1}^i
        with torch.no_grad():
            # 获取服务器模型的设备
            device = next(self.server.model.parameters()).device
            
            for name, param in self.server.model.named_parameters():
                if name in aggregated_delta:
                    # x_t = x_{t-1} - (γ / n) * Σ Δ_i
                    # 注意：aggregated_delta 已经是平均值，所以直接乘以 gamma
                    # 确保 aggregated_delta 在正确的设备上
                    delta_tensor = aggregated_delta[name].to(device)
                    param.data = param.data - gamma * delta_tensor
        
        # 更新全局动量: m_t = (1 / n) * Σ_{i=1}^n m_{t-1,E}^i
        # 将全局动量移到CPU存储（因为服务器模型可能在不同设备上）
        self.global_momentum = {}
        for name, momentum_tensor in aggregated_momentum.items():
            self.global_momentum[name] = momentum_tensor.cpu()

    def averaging(self, models, w=None):
        """
        对模型进行加权平均（与父类保持一致）
        
        Args:
            models: 模型列表
            w: 权重类型（'w' 表示按样本数量加权，None 表示均匀加权）
        
        Returns:
            w_dict: 聚合后的模型参数字典
        """
        if w is None:
            w = self.args.avg
        
        with torch.no_grad():
            # 根据权重类型计算权重
            if w == 'w':
                # 按样本数量加权
                num_sample = torch.tensor([m.train_num for m in models if hasattr(m, 'train_num')])
                if len(num_sample) == 0:
                    # 如果没有 train_num 属性，使用均匀权重
                    weights = 1.0 / len(models)
                else:
                    num_sum = torch.sum(num_sample)
                    if num_sum == 0:
                        weights = 1.0 / len(models)
                    else:
                        weights = num_sample[:, None] / num_sum
            else:
                # 均匀加权
                weights = 1.0 / len(models)
            
            # 获取模型参数形状
            size = {k: v.size() for k, v in models[0].state_dict().items()}
            
            # 将模型转换为特征表示
            feats = self.to_feats(models)
            
            # 将特征展平为向量
            vector = self.feats2vector(feats)
            
            # 计算加权平均
            if isinstance(weights, torch.Tensor):
                w_avg = torch.sum(weights * vector, dim=0)
            else:
                w_avg = torch.mean(vector, dim=0)
            
            # 将向量转换回模型字典
            w_dict = self.vector2model_dict(w_avg, size)
        
        return w_dict
