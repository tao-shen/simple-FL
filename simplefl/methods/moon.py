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


# ==================== 常量定义 ====================

# MOON 默认参数
DEFAULT_MU = 1.0  # 对比损失的权重系数
DEFAULT_TEMPERATURE = 0.5  # 对比损失的温度参数


# ==================== MOON 主类定义 ====================

class MOON(FedAvg):
    """
    MOON: Model-Contrastive Federated Learning
    
    一种用于联邦学习的对比学习方法，旨在通过模型对比学习来增强全局模型的泛化能力。
    该方法通过在每个客户端引入对比损失，鼓励本地模型与全局模型保持一致，
    从而提高模型在异构数据环境下的性能。
    
    核心思想：
    =========
    1. 模型对比学习：
       - 在每个客户端，维护一个全局模型的旧版本（作为负样本）
       - 计算当前本地模型与旧全局模型之间的对比损失
       - 鼓励本地模型的表示与全局模型的表示相似
    
    2. 对比损失函数：
       - 使用基于余弦相似度的对比损失
       - 通过温度参数调节对比损失的敏感度
       - 结合传统的分类损失和对比损失进行训练
    
    3. 减少模型漂移：
       - 通过对比损失，有效地减少客户端模型与全局模型之间的差异
       - 缓解数据异构性带来的模型漂移问题
       - 提高全局模型的泛化能力
    
    工作原理：
    =========
    1. 初始化：服务器初始化全局模型，并将其分发给所有客户端
    
    2. 客户端本地训练：
       a. 保存上一次从服务器接收到的全局模型作为旧模型（负样本）
       b. 在本地训练过程中，对每个 batch：
          - 计算当前模型的输出（正样本）
          - 计算旧模型的输出（负样本）
          - 计算对比损失：鼓励当前模型与旧模型相似
          - 结合分类损失和对比损失更新模型
    
    3. 服务器聚合：
       - 收集所有客户端的模型更新
       - 使用加权平均聚合模型参数
       - 更新全局模型
    
    对比损失公式：
    =============
    L_contrastive = -log(exp(sim(z_local, z_global) / τ) / (exp(sim(z_local, z_global) / τ) + exp(sim(z_local, z_neg) / τ)))
    
    其中：
    - z_local: 当前本地模型的特征表示
    - z_global: 全局模型的特征表示（正样本）
    - z_neg: 旧全局模型的特征表示（负样本）
    - sim: 余弦相似度
    - τ: 温度参数
    
    简化版本（实际实现中常用）：
    L_contrastive = -log(exp(sim(z_local, z_global) / τ) / sum(exp(sim(z_local, z_i) / τ)))
    
    优势：
    =====
    - 减少模型漂移：通过对比学习减少客户端模型与全局模型的差异
    - 提高泛化能力：增强全局模型在未见过的数据上的表现
    - 适应异构数据：在数据异构环境下表现更好
    - 稳定训练过程：减少客户端之间的模型差异，提高训练稳定性
    
    Attributes:
        mu: 对比损失的权重系数，控制对比损失在总损失中的比例
        temperature: 对比损失的温度参数，调节对比损失的敏感度
        old_global_models: 存储每个客户端的旧全局模型
    """
    
    def __init__(self, server, clients, args):
        """
        初始化 MOON
        
        Args:
            server: 服务器对象
            clients: 客户端列表
            args: 配置参数对象
        """
        super().__init__(server, clients, args)
        
        # 从配置中获取 MOON 参数，如果没有则使用默认值
        self.mu = getattr(args, 'moon_mu', DEFAULT_MU)
        self.temperature = getattr(args, 'moon_temperature', DEFAULT_TEMPERATURE)
        
        # 初始化旧全局模型存储（为每个客户端存储上一次的全局模型）
        self.old_global_models = {}
        
        # 打印 MOON 配置信息
        print(f"MOON 初始化完成:")
        print(f"  - 对比损失权重 (mu): {self.mu}")
        print(f"  - 温度参数 (temperature): {self.temperature}")

    def server_init(self):
        """
        服务器初始化：初始化全局模型
        
        与 FedAvg 保持一致，初始化全局模型。
        同时初始化旧全局模型存储。
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()
        
        # 初始化旧全局模型存储（为每个客户端创建初始旧模型）
        for i in range(self.N):
            self.old_global_models[i] = copy.deepcopy(self.server.model)

    def local_update(self, client, model_g, client_idx=None):
        """
        客户端本地更新：执行本地训练并计算对比损失
        
        这是 MOON 的核心方法。在本地训练过程中，需要：
        1. 获取旧全局模型（作为负样本）
        2. 计算当前模型与旧模型的对比损失
        3. 结合分类损失和对比损失进行训练
        
        训练流程：
        =========
        1. 复制全局模型到本地
        2. 获取旧全局模型（如果存在）
        3. 创建优化器
        4. 对每个 epoch：
           a. 对每个 batch：
              - 计算当前模型的输出和损失
              - 计算旧模型的输出（用于对比损失）
              - 计算对比损失
              - 总损失 = 分类损失 + mu * 对比损失
              - 反向传播和参数更新
        
        Args:
            client: 客户端对象，包含本地训练数据
            model_g: 全局模型
            client_idx: 客户端索引（用于获取对应的旧全局模型）
        
        Returns:
            model_l: 本地训练后的模型
        """
        # 复制全局模型到本地
        model_l = copy.deepcopy(model_g)
        
        # 获取旧全局模型（作为负样本）
        if client_idx is not None and client_idx in self.old_global_models:
            old_model_g = self.old_global_models[client_idx]
        else:
            # 如果没有旧模型，使用当前全局模型
            old_model_g = copy.deepcopy(model_g)
        
        # 将旧模型设置为评估模式（不需要梯度）
        old_model_g.eval()
        
        # 创建优化器
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), 
            lr=self.args.lr_l, 
            weight_decay=self.args.weight_decay
        )
        
        # 准备数据加载器
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), 
                batch_size=self.args.local_batch_size, 
                shuffle=True
            )
        except:  # 跳过没有数据的客户端
            return model_l
        
        # 将模型移到设备上
        model_l = model_l.to(self.args.device)
        old_model_g = old_model_g.to(self.args.device)
        
        # 执行本地训练
        for epoch in range(self.E):
            model_l.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # 将 batch 移到设备上
                batch = to_device(batch, self.args.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播：计算当前模型的输出
                output_local = model_l(batch)
                labels = batch["label"]
                
                # 计算分类损失
                loss_cls = model_l.loss_fn(output_local, labels)
                
                # 计算对比损失
                # 使用 no_grad 来避免计算旧模型的梯度
                with torch.no_grad():
                    output_old = old_model_g(batch)
                
                # 计算对比损失
                contrastive_loss = self._compute_contrastive_loss(
                    output_local, output_old
                )
                
                # 总损失 = 分类损失 + mu * 对比损失
                total_loss = loss_cls + self.mu * contrastive_loss
                
                # 反向传播
                total_loss.backward()
                
                # 更新参数
                optimizer.step()
        
        # 将模型移回 CPU 以节省内存
        model_l = model_l.cpu()
        
        return model_l

    def _compute_contrastive_loss(self, output_local, output_old):
        """
        计算对比损失
        
        使用基于余弦相似度的对比损失，鼓励本地模型与旧全局模型的表示相似。
        
        MOON 的对比损失公式：
        ====================
        L_contrastive = -log(exp(sim(z_local, z_global) / τ) / (exp(sim(z_local, z_global) / τ) + exp(sim(z_local, z_old) / τ)))
        
        其中：
        - z_local: 当前本地模型的输出（特征表示）
        - z_global: 全局模型的输出（正样本，在当前实现中我们使用 z_local 作为正样本）
        - z_old: 旧全局模型的输出（负样本）
        - sim: 余弦相似度
        - τ: 温度参数
        
        简化版本（实际实现中常用）：
        ===========================
        由于在本地训练时，我们只有旧全局模型作为参考，我们使用简化的对比损失：
        L_contrastive = -log(exp(sim(z_local, z_old) / τ) / (exp(sim(z_local, z_old) / τ) + 1))
        
        或者更简单的形式（当只有一个负样本时）：
        L_contrastive = -sim(z_local, z_old) / τ
        
        我们使用负余弦相似度作为损失，因为要最大化相似度（最小化负相似度）。
        
        Args:
            output_local: 当前本地模型的输出，形状为 (batch_size, num_classes)
            output_old: 旧全局模型的输出，形状为 (batch_size, num_classes)
        
        Returns:
            contrastive_loss: 对比损失值（标量）
        """
        # 对输出进行 L2 归一化（用于计算余弦相似度）
        # 归一化后的向量可以用于计算余弦相似度
        output_local_norm = F.normalize(output_local, p=2, dim=1)
        output_old_norm = F.normalize(output_old, p=2, dim=1)
        
        # 计算余弦相似度：sim = z_local · z_old / (||z_local|| * ||z_old||)
        # 由于已经归一化，所以 sim = z_local · z_old
        # 形状: (batch_size,)
        cosine_sim = (output_local_norm * output_old_norm).sum(dim=1)
        
        # 计算对比损失
        # 方法1：使用 InfoNCE 损失（更符合 MOON 原始实现）
        # 但由于我们只有一个负样本，可以简化为：
        # L = -log(exp(sim / τ) / (exp(sim / τ) + 1))
        # 这等价于：L = -log(sigmoid(sim / τ))
        # 或者更简单的形式：L = -sim / τ（当 sim 接近 1 时）
        
        # 方法2：直接使用负余弦相似度（简化版本，更稳定）
        # 我们使用负相似度作为损失，因为要最大化相似度
        # 除以温度参数 τ 来调节损失的尺度
        contrastive_loss = -cosine_sim.mean() / self.temperature
        
        # 方法3：使用 InfoNCE 损失（如果需要更精确的实现）
        # sim_scaled = cosine_sim / self.temperature
        # contrastive_loss = -F.logsigmoid(sim_scaled).mean()
        
        return contrastive_loss

    def clients_update(self):
        """
        客户端更新：收集所有候选客户端的模型更新
        
        遍历所有候选客户端，执行本地训练。
        在训练前，需要为每个客户端设置对应的旧全局模型。
        """
        self.models = []
        
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # 执行本地训练（传入客户端索引以获取对应的旧全局模型）
            model = self.local_update(self.clients[k], self.server.model, client_idx=k)
            self.models.append(model.cpu())

    def server_update(self):
        """
        服务器更新：聚合客户端更新并更新全局模型和旧全局模型
        
        在聚合后，需要更新每个客户端的旧全局模型，以便下一轮使用。
        
        更新流程：
        =========
        1. 聚合客户端模型更新
        2. 更新全局模型
        3. 更新每个客户端的旧全局模型（保存当前全局模型作为下一轮的旧模型）
        """
        # 使用标准的加权平均聚合模型
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)
        
        # 更新每个客户端的旧全局模型
        # 将当前全局模型保存为下一轮的旧模型
        for k in self.candidates:
            self.old_global_models[k] = copy.deepcopy(self.server.model)

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
