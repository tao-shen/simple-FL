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
from typing import Optional, Dict, Tuple


# ==================== 常量定义 ====================

# FedLion 默认参数
DEFAULT_BETA1 = 0.9  # Lion 优化器的动量系数
DEFAULT_BETA2 = 0.99  # Lion 优化器的二阶动量系数（如果使用）
DEFAULT_WEIGHT_DECAY = 0.0  # 权重衰减系数


# ==================== Lion 优化器类定义 ====================

class Lion(torch.optim.Optimizer):
    """
    Lion (EvoLved Sign Momentum) 优化器
    
    Lion 是一种基于符号梯度的优化器，通过使用梯度的符号而不是原始梯度值来更新参数。
    这种方法具有以下优势：
    1. 减少内存使用：只需要存储符号，而不是完整的梯度值
    2. 提高数值稳定性：符号操作对梯度缩放不敏感
    3. 减少通信开销：在联邦学习中，只需要传输符号梯度
    
    Lion 更新规则：
    ==============
    u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
    w_t = w_{t-1} - lr * sign(u_t)
    
    其中：
    - g_t: 当前梯度
    - sign(g_t): 梯度的符号（+1, 0, 或 -1）
    - u_t: 动量项
    - β1: 动量系数（通常为 0.9）
    - lr: 学习率
    - w_t: 更新后的参数
    
    与 Adam 的区别：
    ===============
    - Adam 使用原始梯度值和二阶矩估计
    - Lion 只使用梯度的符号，不需要二阶矩估计
    - Lion 更简单，内存效率更高
    
    Attributes:
        beta1: 动量系数
        weight_decay: 权重衰减系数
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        初始化 Lion 优化器
        
        Args:
            params: 要优化的参数（通常是 model.parameters()）
            lr: 学习率（默认 1e-4）
            betas: 动量系数元组 (beta1, beta2)
                   - beta1: 一阶动量系数（默认 0.9）
                   - beta2: 二阶动量系数（在 Lion 中通常不使用，但保留以保持接口一致性）
            weight_decay: 权重衰减系数（默认 0.0）
        """
        if not 0.0 <= lr:
            raise ValueError(f"学习率必须非负: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 必须在 [0, 1) 范围内: {betas[0]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"权重衰减必须非负: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)
        
        # 保存参数
        self.beta1 = betas[0]
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步优化
        
        Lion 更新规则：
        1. 计算符号梯度：sign(g_t)
        2. 更新动量：u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
        3. 更新参数：w_t = w_{t-1} - lr * sign(u_t)
        
        Args:
            closure: 一个可调用对象，用于重新计算损失和梯度（可选）
        
        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                lr = group['lr']
                beta1 = group['betas'][0]
                weight_decay = group['weight_decay']
                
                # 获取或初始化动量状态
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                
                # 计算符号梯度：sign(g_t)
                sign_grad = torch.sign(grad)
                
                # 更新动量：u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
                exp_avg.mul_(beta1).add_(sign_grad, alpha=1 - beta1)
                
                # 应用权重衰减（如果启用）
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # 更新参数：w_t = w_{t-1} - lr * sign(u_t)
                # Lion 优化器的关键：使用 sign(u_t) 而不是 u_t 本身
                p.data.add_(torch.sign(exp_avg), alpha=-lr)
        
        return loss


# ==================== FedLion 主类定义 ====================

class FedLion(FedAvg):
    """
    FedLion: Federated Lion Optimizer（联邦 Lion 优化器）
    
    一种自适应的联邦优化算法，旨在提高联邦学习的收敛速度并减少通信开销。
    FedLion 将 Lion 优化器的关键元素无缝集成到联邦学习框架中。
    
    核心思想：
    =========
    1. Lion 优化器：
       - 使用符号梯度（sign-based gradient）而不是原始梯度
       - 自适应学习率和动量机制
       - 减少内存使用和通信开销
    
    2. 联邦学习集成：
       - 在客户端本地使用 Lion 优化器进行训练
       - 在服务器端聚合模型更新
       - 利用符号梯度减少通信开销
    
    3. 优势：
       - 更快的收敛速度：自适应学习率机制
       - 减少通信开销：只传输符号梯度或模型更新
       - 更好的泛化性能：Lion 优化器的特性
    
    工作原理：
    =========
    1. 初始化：服务器初始化全局模型，并将其分发给所有客户端
    
    2. 客户端本地训练：
       a. 复制全局模型到本地
       b. 创建 Lion 优化器
       c. 在本地数据上进行训练：
          - 计算梯度
          - 使用符号梯度更新参数
          - Lion 优化器自动处理动量和学习率
    
    3. 服务器聚合：
       - 收集所有客户端的模型更新
       - 使用加权平均聚合模型参数
       - 更新全局模型
    
    Lion 优化器更新规则：
    ====================
    u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
    w_t = w_{t-1} - lr * sign(u_t)
    
    其中：
    - g_t: 当前梯度
    - sign(g_t): 梯度的符号（+1, 0, 或 -1）
    - u_t: 动量项
    - β1: 动量系数（通常为 0.9）
    - lr: 学习率
    - w_t: 更新后的参数
    
    优势：
    =====
    - 更快的收敛速度：自适应学习率机制
    - 减少通信开销：符号梯度只需要 1 bit 信息
    - 更好的数值稳定性：符号操作对梯度缩放不敏感
    - 内存效率：不需要存储完整的梯度值
    
    Attributes:
        beta1: Lion 优化器的动量系数
        beta2: Lion 优化器的二阶动量系数（保留以保持接口一致性）
        weight_decay: 权重衰减系数
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
        self.weight_decay = getattr(args, 'lion_weight_decay', DEFAULT_WEIGHT_DECAY)
        
        # 打印 FedLion 配置信息
        print(f"FedLion 初始化完成:")
        print(f"  - 动量系数 (beta1): {self.beta1}")
        print(f"  - 二阶动量系数 (beta2): {self.beta2}")
        print(f"  - 权重衰减 (weight_decay): {self.weight_decay}")

    def server_init(self):
        """
        服务器初始化：初始化全局模型
        
        与 FedAvg 保持一致，初始化全局模型。
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()

    def local_update(self, client, model_g):
        """
        客户端本地更新：使用 Lion 优化器进行本地训练
        
        这是 FedLion 的核心方法。在本地训练过程中，使用 Lion 优化器代替传统的 SGD 或 Adam。
        Lion 优化器使用符号梯度进行更新，具有更好的数值稳定性和更低的通信开销。
        
        训练流程：
        =========
        1. 复制全局模型到本地
        2. 创建 Lion 优化器
        3. 对每个 epoch：
           a. 对每个 batch：
              - 前向传播：计算预测值和损失
              - 反向传播：计算梯度
              - Lion 优化器更新：
                * 计算符号梯度：sign(gradient)
                * 更新动量：u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
                * 更新参数：w_t = w_{t-1} - lr * sign(u_t)
        
        Args:
            client: 客户端对象，包含本地训练数据
            model_g: 全局模型
        
        Returns:
            model_l: 本地训练后的模型
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
            return model_l
        
        # 将模型移动到正确的设备上
        model_l = model_l.to(self.args.device)
        
        # 创建 Lion 优化器
        # Lion 优化器使用符号梯度，具有更好的数值稳定性和更低的通信开销
        optimizer = Lion(
            model_l.parameters(),
            lr=self.args.lr_l,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay if self.weight_decay > 0 else self.args.weight_decay
        )
        
        # 执行本地训练
        for epoch in range(self.E):
            model_l.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # 将 batch 移到设备上
                batch = to_device(batch, self.args.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播：计算预测值
                output = model_l(batch)
                labels = batch["label"]
                
                # 计算损失
                loss = model_l.loss_fn(output, labels)
                
                # 反向传播：计算梯度
                loss.backward()
                
                # Lion 优化器更新参数
                # Lion 内部会：
                # 1. 计算符号梯度：sign(gradient)
                # 2. 更新动量：u_t = β1 * u_{t-1} + (1 - β1) * sign(g_t)
                # 3. 更新参数：w_t = w_{t-1} - lr * sign(u_t)
                optimizer.step()
        
        # 将模型移回 CPU 以节省内存
        model_l = model_l.cpu()
        
        return model_l

    def clients_update(self):
        """
        客户端更新：收集所有候选客户端的模型更新
        
        遍历所有候选客户端，使用 Lion 优化器进行本地训练。
        """
        self.models = []
        
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # 使用 Lion 优化器进行本地训练
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())

    def server_update(self):
        """
        服务器更新：聚合客户端更新并更新全局模型
        
        使用标准的加权平均聚合模型参数。
        
        更新流程：
        =========
        1. 聚合客户端模型更新
        2. 更新全局模型
        """
        # 使用标准的加权平均聚合模型
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)

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
