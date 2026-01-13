from .fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch.utils.data import DataLoader


# ==================== 常量定义 ====================

# FedNova 默认参数
DEFAULT_NORMALIZATION_METHOD = 'linear'  # 归一化方法：'linear' 或 'constant'
DEFAULT_MOMENTUM = 0.0  # 动量系数（如果使用动量归一化）


# ==================== FedNova 主类定义 ====================

class FedNova(FedAvg):
    """
    FedNova: 联邦归一化平均方法
    
    针对联邦学习中客户端数据和计算资源异构性问题的优化算法。
    在传统的 FedAvg 方法中，各客户端在本地进行多轮训练后，将模型更新发送至服务器进行加权平均。
    然而，由于各客户端的数据量和计算能力不同，导致本地训练的轮数存在差异，
    这可能引发全局模型目标函数的不一致性，影响收敛性能。
    
    核心思想：
    =========
    FedNova 通过对本地更新进行归一化处理，消除由于客户端本地训练轮数差异导致的
    目标函数不一致性。具体而言，FedNova 在聚合客户端更新时，对每个客户端的更新量
    进行归一化，使得每个客户端的贡献与其本地训练轮数成比例，从而确保全局模型的
    优化方向与真实目标函数一致。
    
    归一化方法：
    ===========
    1. Linear Normalization（线性归一化）：
       - delta_normalized = delta / E_i
       - 其中 E_i 是客户端 i 的本地训练轮数
       - 这是最常用的归一化方法
    
    2. Constant Normalization（常数归一化）：
       - delta_normalized = delta / E_max
       - 其中 E_max 是所有客户端的最大训练轮数
       - 适用于所有客户端使用相同训练轮数的情况
    
    优势：
    =====
    - 消除目标函数不一致性：通过归一化处理，确保全局优化方向正确
    - 提高收敛速度：减少由于异构性导致的收敛延迟
    - 改善模型性能：在异构环境下获得更好的最终性能
    - 适应性强：可以处理不同数据量和计算能力的客户端
    
    Attributes:
        normalization_method: 归一化方法（'linear' 或 'constant'）
        local_epochs_list: 记录每个客户端的本地训练轮数
    """
    
    def __init__(self, server, clients, args):
        """
        初始化 FedNova
        
        Args:
            server: 服务器对象
            clients: 客户端列表
            args: 配置参数对象
        """
        super().__init__(server, clients, args)
        
        # 从配置中获取归一化方法，如果没有则使用默认值
        self.normalization_method = getattr(args, 'nova_normalization', DEFAULT_NORMALIZATION_METHOD)
        
        # 初始化本地训练轮数列表（用于记录每个客户端的训练轮数）
        self.local_epochs_list = []
        
        # 打印 FedNova 配置信息
        print(f"FedNova 初始化完成:")
        print(f"  - 归一化方法: {self.normalization_method}")

    def server_init(self):
        """
        服务器初始化：初始化全局模型
        
        与 FedAvg 保持一致，初始化全局模型。
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()

    def local_update(self, client, model_g):
        """
        客户端本地更新：执行本地训练并记录训练轮数
        
        这是 FedNova 的核心方法之一。在本地训练过程中，需要记录实际执行的训练轮数，
        以便在服务器端进行归一化处理。
        
        训练流程：
        =========
        1. 复制全局模型到本地
        2. 创建优化器
        3. 执行 E 轮本地训练
        4. 返回训练后的模型
        
        注意：这里记录的是实际执行的训练轮数，可能与配置的 E 不同
        （例如，如果客户端数据为空，则训练轮数为 0）
        
        Args:
            client: 客户端对象，包含本地训练数据
            model_g: 全局模型
        
        Returns:
            model_l: 本地训练后的模型
        """
        # 复制全局模型到本地
        model_l = copy.deepcopy(model_g)
        
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
            # 如果没有数据，记录训练轮数为 0
            return model_l
        
        # 记录实际执行的训练轮数
        actual_epochs = 0
        
        # 执行本地训练
        for E in range(self.E):
            # 执行一轮训练
            model_l.fit(train_loader, optimizer)
            actual_epochs += 1
        
        # 将实际训练轮数存储到模型中，以便服务器端使用
        # 注意：这里使用模型的 train_num 属性来存储训练轮数
        # 如果模型没有 train_num 属性，我们会在 clients_update 中单独记录
        if not hasattr(model_l, 'train_num'):
            model_l.train_num = actual_epochs
        
        return model_l

    def clients_update(self):
        """
        客户端更新：收集所有候选客户端的模型更新和训练轮数
        
        遍历所有候选客户端，执行本地训练，并记录每个客户端的训练轮数。
        这些信息将用于服务器端的归一化聚合。
        
        与 FedAvg 不同，这里需要额外记录每个客户端的训练轮数。
        """
        self.models = []
        self.local_epochs_list = []  # 清空之前的记录
        
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # 执行本地训练
            model = self.local_update(self.clients[k], self.server.model)
            
            # 记录训练轮数
            # 优先使用模型的 train_num 属性，如果没有则使用配置的 E
            if hasattr(model, 'train_num'):
                epochs = model.train_num
            else:
                epochs = self.E
            
            self.local_epochs_list.append(epochs)
            self.models.append(model.cpu())

    def server_update(self):
        """
        服务器更新：使用归一化平均聚合客户端更新
        
        这是 FedNova 的核心方法。在聚合客户端更新时，对每个客户端的更新量进行归一化处理，
        消除由于本地训练轮数差异导致的目标函数不一致性。
        
        归一化聚合流程：
        ===============
        1. 计算每个客户端的模型更新量：delta_i = w_i - w_global
        2. 对每个客户端的更新进行归一化：delta_normalized_i = delta_i / E_i
           - E_i 是客户端 i 的本地训练轮数
        3. 对归一化后的更新进行加权平均
        4. 更新全局模型：w_global = w_global + aggregated_delta
        
        归一化方法说明：
        ===============
        - Linear Normalization: delta_normalized = delta / E_i
          - 每个客户端的更新量除以其本地训练轮数
          - 确保每个客户端的贡献与其训练轮数成比例
        
        - Constant Normalization: delta_normalized = delta / E_max
          - 所有客户端的更新量除以最大训练轮数
          - 适用于所有客户端使用相同训练轮数的情况
        
        聚合公式：
        =========
        w_global = w_global + sum(n_i * delta_normalized_i) / sum(n_i)
        
        其中：
        - n_i: 客户端 i 的样本数量（如果使用加权平均）
        - delta_normalized_i: 客户端 i 的归一化更新量
        """
        # 检查是否有模型和训练轮数记录
        if len(self.models) == 0:
            return
        
        # 确保训练轮数记录完整
        if len(self.local_epochs_list) != len(self.models):
            # 如果训练轮数记录不完整，使用配置的 E 作为默认值
            self.local_epochs_list = [self.E] * len(self.models)
        
        # 获取全局模型参数
        w_global = self.server.model.state_dict()
        
        # 计算归一化后的更新量总和
        aggregated_delta = {}
        total_weight = 0.0
        
        # 计算最大训练轮数（用于常数归一化）
        max_epochs = max(self.local_epochs_list) if self.local_epochs_list else self.E
        
        for i, model in enumerate(self.models):
            # 获取客户端的训练轮数
            epochs = self.local_epochs_list[i]
            
            # 如果训练轮数为 0，跳过该客户端
            if epochs == 0:
                continue
            
            # 获取客户端权重（样本数量或均匀权重）
            if hasattr(model, 'train_num') and self.args.avg == 'w':
                weight = model.train_num
            else:
                weight = 1.0
            
            # 计算模型更新量：delta = w_local - w_global
            w_local = model.state_dict()
            
            # 对每个参数计算归一化更新量
            for key in w_global.keys():
                if key not in aggregated_delta:
                    aggregated_delta[key] = torch.zeros_like(w_global[key])
                
                # 计算更新量
                delta = w_local[key] - w_global[key]
                
                # 对更新量进行归一化
                if self.normalization_method == 'linear':
                    # 线性归一化：除以本地训练轮数
                    normalized_delta = delta / epochs
                elif self.normalization_method == 'constant':
                    # 常数归一化：除以最大训练轮数
                    normalized_delta = delta / max_epochs
                else:
                    # 默认使用线性归一化
                    normalized_delta = delta / epochs
                
                # 累加加权归一化更新量
                aggregated_delta[key] += weight * normalized_delta
            
            # 累加总权重
            total_weight += weight
        
        # 如果总权重为 0，则不更新
        if total_weight == 0:
            return
        
        # 计算加权平均的归一化更新量
        for key in aggregated_delta.keys():
            aggregated_delta[key] /= total_weight
        
        # 更新全局模型：w_global = w_global + aggregated_delta
        w_new = {}
        for key in w_global.keys():
            w_new[key] = w_global[key] + aggregated_delta[key]
        
        # 加载更新后的模型参数
        self.server.model.load_state_dict(w_new)

    def _compute_delta(self, model_l, model_g):
        """
        计算模型更新量：本地模型与全局模型的差值
        
        用于计算客户端本地训练后的模型更新量。
        
        Args:
            model_l: 本地模型
            model_g: 全局模型
        
        Returns:
            delta_model: 模型更新量（本地模型 - 全局模型）
        """
        # 将模型移到 CPU 以节省内存
        model_l = model_l.cpu()
        
        # 获取模型参数字典
        w_l = model_l.state_dict()
        w_g = model_g.state_dict()
        
        # 创建更新量字典
        delta_dict = {}
        for key in w_l.keys():
            # 计算参数差值：delta = w_l - w_g
            delta_dict[key] = w_l[key] - w_g[key]
        
        return delta_dict

    def _normalize_delta(self, delta_dict, epochs):
        """
        对模型更新量进行归一化处理
        
        将更新量除以其对应的训练轮数，实现归一化。
        
        Args:
            delta_dict: 模型更新量字典
            epochs: 训练轮数（用于归一化）
        
        Returns:
            normalized_delta: 归一化后的更新量字典
        """
        normalized_delta = {}
        
        # 如果训练轮数为 0，返回零更新量
        if epochs == 0:
            for key in delta_dict.keys():
                normalized_delta[key] = torch.zeros_like(delta_dict[key])
            return normalized_delta
        
        # 对每个参数进行归一化：delta_normalized = delta / epochs
        for key in delta_dict.keys():
            normalized_delta[key] = delta_dict[key] / epochs
        
        return normalized_delta

    def _apply_delta(self, model_g, delta_dict):
        """
        将更新量应用到全局模型上
        
        用于创建归一化后的模型，以便进行聚合。
        
        Args:
            model_g: 全局模型
            delta_dict: 更新量字典
        
        Returns:
            updated_model: 应用更新后的模型
        """
        # 复制全局模型
        updated_model = copy.deepcopy(model_g)
        
        # 获取更新后模型的参数字典
        w_g = model_g.state_dict()
        w_updated = {}
        
        # 应用更新量：w_updated = w_g + delta
        for key in w_g.keys():
            w_updated[key] = w_g[key] + delta_dict[key]
        
        # 加载更新后的参数
        updated_model.load_state_dict(w_updated)
        
        return updated_model

    def averaging(self, models, w=None):
        """
        对模型进行加权平均（重写父类方法以保持兼容性）
        
        这个方法与父类 FedAvg 的 averaging 方法保持一致，
        但在这里用于聚合归一化后的模型。
        
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
                    weights = torch.ones(len(models)) / len(models)
                else:
                    num_sum = torch.sum(num_sample)
                    if num_sum == 0:
                        weights = torch.ones(len(models)) / len(models)
                    else:
                        weights = num_sample[:, None] / num_sum
            else:
                # 均匀加权
                weights = torch.ones(len(models)) / len(models)
            
            # 获取模型参数形状
            size = {k: v.size() for k, v in models[0].state_dict().items()}
            
            # 将模型转换为特征表示
            feats = self.to_feats(models)
            
            # 将特征展平为向量
            vector = self.feats2vector(feats)
            
            # 计算加权平均
            w_avg = torch.sum(weights * vector, dim=0)
            
            # 将向量转换回模型字典
            w_dict = self.vector2model_dict(w_avg, size)
        
        return w_dict
