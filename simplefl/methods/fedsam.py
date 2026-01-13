from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional


# ==================== 常量定义 ====================

# SAM 优化器默认参数
DEFAULT_RHO = 0.05  # SAM 扰动半径，控制扰动的大小
DEFAULT_ADAPTIVE = False  # 是否使用自适应扰动半径


# ==================== SAM 优化器类定义 ====================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) 优化器
    
    SAM 是一种寻找平坦最小值的优化方法，通过在参数空间中沿梯度方向进行扰动，
    然后基于扰动后的损失进行梯度更新，从而找到更平坦的损失最小值。
    平坦的最小值通常具有更好的泛化能力。
    
    核心思想：
    =========
    1. 第一步：在参数空间中沿梯度方向进行扰动
       w_perturbed = w + rho * normalize(gradient)
    
    2. 第二步：计算扰动后的损失和梯度
       loss_perturbed = loss(w_perturbed)
       gradient_perturbed = grad(loss_perturbed)
    
    3. 第三步：基于扰动后的梯度更新原始参数
       w_new = w - lr * gradient_perturbed
    
    这种方法能够帮助模型找到更平坦的最小值，提高泛化能力。
    
    Attributes:
        base_optimizer: 基础优化器（如 SGD 或 Adam）
        rho: 扰动半径，控制扰动的大小
        adaptive: 是否使用自适应扰动半径
    """
    
    def __init__(self, params, base_optimizer, rho=DEFAULT_RHO, adaptive=DEFAULT_ADAPTIVE, **kwargs):
        """
        初始化 SAM 优化器
        
        Args:
            params: 要优化的参数（通常是 model.parameters()）
            base_optimizer: 基础优化器类（如 torch.optim.SGD 或 torch.optim.Adam）
            rho: 扰动半径，控制扰动的大小（默认 0.05）
            adaptive: 是否使用自适应扰动半径（默认 False）
            **kwargs: 传递给基础优化器的其他参数（如学习率 lr）
        """
        # 检查扰动半径是否非负
        assert rho >= 0.0, f"扰动半径 rho 必须非负，当前值: {rho}"
        
        # 设置默认参数
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # 初始化基础优化器
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
        # 保存扰动半径和自适应标志
        self.rho = rho
        self.adaptive = adaptive
        
        # 保存扰动信息，用于 second_step 中恢复参数
        self.perturbation_info = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        SAM 第一步：在参数空间中沿梯度方向进行扰动
        
        这一步会修改模型参数，将参数移动到扰动位置：
        w_perturbed = w + rho * normalize(gradient)
        
        注意：这一步会修改原始参数，需要在 second_step 中恢复。
        
        Args:
            zero_grad: 是否在扰动后清零梯度（默认 False）
        """
        # 计算所有参数的梯度范数
        grad_norm = self._grad_norm()
        
        # 清空之前的扰动信息
        self.perturbation_info = {}
        
        # 对每个参数组进行处理
        for group_idx, group in enumerate(self.param_groups):
            # 获取该参数组的扰动半径
            scale = group["rho"] / (grad_norm + 1e-12)
            
            # 保存扰动信息，用于 second_step 中恢复参数
            self.perturbation_info[group_idx] = {
                'scale': scale,
                'perturbations': []
            }
            
            # 对每个参数进行扰动
            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                # 计算扰动向量：沿梯度方向移动
                # 对于自适应 SAM，使用参数的平方作为权重
                adaptive_weight = torch.pow(p, 2) if self.adaptive else 1.0
                e_w = adaptive_weight * p.grad * scale.to(p)
                # 保存扰动向量，用于 second_step 中恢复
                self.perturbation_info[group_idx]['perturbations'].append(e_w.clone())
                # 应用扰动：w_perturbed = w + rho * normalize(gradient)
                p.add_(e_w)
        
        # 如果需要，清零梯度
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        SAM 第二步：恢复原始参数并执行基础优化器的更新
        
        这一步会：
        1. 恢复原始参数（撤销 first_step 中的扰动）
        2. 使用基础优化器（如 SGD 或 Adam）更新参数
        3. 更新基于扰动后的梯度（在 first_step 后重新计算的梯度）
        
        Args:
            zero_grad: 是否在更新后清零梯度（默认 False）
        """
        # 对每个参数组进行处理
        for group_idx, group in enumerate(self.param_groups):
            # 获取该参数组对应的扰动信息
            if group_idx not in self.perturbation_info:
                continue
            
            perturbations = self.perturbation_info[group_idx]['perturbations']
            
            # 对每个参数恢复原始位置（撤销扰动）
            for param_idx, p in enumerate(group["params"]):
                if param_idx >= len(perturbations):
                    continue
                # 恢复原始参数：w = w_perturbed - e_w
                # 使用保存的扰动向量来恢复，确保使用相同的扰动值
                e_w = perturbations[param_idx]
                p.sub_(e_w)
        
        # 使用基础优化器更新参数（基于扰动后的梯度）
        self.base_optimizer.step()
        
        # 如果需要，清零梯度
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        SAM 优化器的完整更新步骤
        
        这个方法会执行 SAM 的两步更新过程：
        1. first_step: 在参数空间中沿梯度方向进行扰动
        2. 重新计算损失和梯度（通过 closure）
        3. second_step: 恢复原始参数并使用扰动后的梯度更新
        
        Args:
            closure: 一个可调用对象，用于重新计算损失和梯度
                    必须提供，因为需要在扰动后重新计算梯度
        
        Returns:
            损失值（如果提供了 closure）
        """
        # 检查是否提供了 closure
        assert closure is not None, "SAM 优化器需要一个 closure 函数来重新计算损失和梯度"
        
        # 第一步：在参数空间中沿梯度方向进行扰动
        self.first_step(zero_grad=True)
        
        # 重新计算损失和梯度（在扰动后的参数上）
        # closure 应该返回损失值，并计算新的梯度
        closure = torch.enable_grad()(closure)
        loss = closure()
        
        # 第二步：恢复原始参数并使用扰动后的梯度更新
        self.second_step()
        
        return loss

    def _grad_norm(self):
        """
        计算所有参数的梯度范数
        
        用于归一化扰动向量，确保扰动的大小由 rho 控制。
        
        Returns:
            所有参数梯度的 L2 范数
        """
        # 收集所有参数的梯度
        grad_norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # 计算该参数的梯度范数
                    if self.adaptive:
                        grad_norm = (torch.abs(p) * p.grad).norm(p=2)
                    else:
                        grad_norm = p.grad.norm(p=2)
                    grad_norms.append(grad_norm)
        
        # 如果没有梯度，返回零
        if len(grad_norms) == 0:
            # 返回一个在正确设备上的零张量
            if len(self.param_groups) > 0 and len(self.param_groups[0]["params"]) > 0:
                device = self.param_groups[0]["params"][0].device
                return torch.tensor(0.0, device=device)
            else:
                return torch.tensor(0.0)
        
        # 将所有梯度范数堆叠并计算总范数
        shared_device = grad_norms[0].device
        norm = torch.norm(
            torch.stack([g.to(shared_device) for g in grad_norms]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        """
        加载优化器状态
        
        Args:
            state_dict: 状态字典
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ==================== FedSAM 主类定义 ====================

class FedSAM(FedAvgM):
    """
    FedSAM: 联邦 Sharpness-Aware Minimization 方法
    
    在联邦学习框架中应用 SAM 优化器，通过在客户端本地训练时使用 SAM 优化器
    来寻找平坦的最小值，从而提高模型的泛化能力和鲁棒性。
    
    核心思想：
    =========
    1. 客户端本地训练：使用 SAM 优化器代替传统的 SGD 或 Adam
       - SAM 会在参数空间中寻找平坦的最小值
       - 平坦的最小值对数据分布变化更鲁棒，适合联邦学习中的非 IID 场景
    
    2. 服务器聚合：使用标准的 FedAvgM 聚合方式
       - 收集客户端更新后的模型
       - 使用动量机制聚合模型更新
    
    优势：
    =====
    - 提高模型泛化能力：平坦的最小值通常具有更好的泛化性能
    - 适应数据异质性：对客户端数据分布差异更鲁棒
    - 减少过拟合：通过寻找平坦最小值，减少对特定数据分布的过拟合
    
    Attributes:
        sam_rho: SAM 扰动半径，控制扰动的大小
        sam_adaptive: 是否使用自适应扰动半径
    """
    
    def __init__(self, server, clients, args):
        """
        初始化 FedSAM
        
        Args:
            server: 服务器对象
            clients: 客户端列表
            args: 配置参数对象
        """
        super().__init__(server, clients, args)
        
        # 从配置中获取 SAM 参数，如果没有则使用默认值
        self.sam_rho = getattr(args, 'sam_rho', DEFAULT_RHO)
        self.sam_adaptive = getattr(args, 'sam_adaptive', DEFAULT_ADAPTIVE)
        
        # 打印 SAM 配置信息
        print(f"FedSAM 初始化完成:")
        print(f"  - SAM 扰动半径 (rho): {self.sam_rho}")
        print(f"  - 自适应扰动: {self.sam_adaptive}")

    def server_init(self):
        """
        服务器初始化：初始化全局模型和动量缓冲区
        
        与 FedAvgM 保持一致，初始化全局模型和用于动量的缓冲区。
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()
        
        # 初始化动量缓冲区（用于 FedAvgM 的动量更新）
        m = copy.deepcopy(self.server.model)
        self.zero_weights(m)
        self.m = m.state_dict()

    def local_update(self, client, model_g):
        """
        客户端本地更新：使用 SAM 优化器进行本地训练
        
        这是 FedSAM 的核心方法，使用 SAM 优化器代替传统的优化器进行本地训练。
        SAM 优化器会寻找平坦的最小值，从而提高模型的泛化能力。
        
        训练流程：
        =========
        1. 复制全局模型到本地
        2. 创建 SAM 优化器（基于 SGD 或 Adam）
        3. 对每个 epoch：
           a. 对每个 batch：
              - SAM first_step: 在参数空间中沿梯度方向进行扰动
              - 重新计算损失和梯度（在扰动后的参数上）
              - SAM second_step: 恢复原始参数并使用扰动后的梯度更新
        
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
        
        # 选择基础优化器类型（SGD 或 Adam）
        # 从父类继承的 opts 字典中获取优化器类，如果没有则使用默认值
        if hasattr(self, 'opts') and self.args.local_optimizer in self.opts:
            base_optimizer_class = self.opts[self.args.local_optimizer]
        else:
            # 默认使用 SGD 或 Adam
            if self.args.local_optimizer == 'adam':
                base_optimizer_class = torch.optim.Adam
            else:
                base_optimizer_class = torch.optim.SGD
        
        # 创建 SAM 优化器
        # SAM 需要一个基础优化器（如 SGD 或 Adam）和扰动半径 rho
        sam_optimizer = SAM(
            model_l.parameters(),
            base_optimizer=base_optimizer_class,
            rho=self.sam_rho,
            adaptive=self.sam_adaptive,
            lr=self.args.lr_l,
            weight_decay=self.args.weight_decay
        )
        
        # 本地训练多个 epoch
        for epoch in range(self.E):
            model_l.train()
            
            # 对每个 batch 进行训练
            for batch_idx, batch in enumerate(train_loader):
                # 将 batch 移到设备上
                batch = to_device(batch, self.args.device)
                
                # 定义 closure 函数：计算损失和梯度
                def closure():
                    """
                    SAM 优化器需要的 closure 函数
                    
                    这个函数会在扰动后的参数上重新计算损失和梯度。
                    SAM 优化器会调用这个函数两次：
                    1. 第一次：在原始参数上计算梯度（用于 first_step 的扰动）
                    2. 第二次：在扰动后的参数上计算梯度（用于 second_step 的更新）
                    """
                    # 清零梯度
                    sam_optimizer.zero_grad()
                    
                    # 前向传播：计算预测值
                    predictions = model_l(batch)
                    
                    # 获取标签
                    labels = batch["label"]
                    
                    # 计算损失
                    loss = model_l.loss_fn(predictions, labels)
                    
                    # 反向传播：计算梯度
                    loss.backward()
                    
                    return loss
                
                # 使用 SAM 优化器更新参数
                # SAM 的 step 方法会：
                # 1. 调用 first_step 进行扰动
                # 2. 调用 closure 重新计算损失和梯度
                # 3. 调用 second_step 恢复参数并更新
                sam_optimizer.step(closure)
        
        # 将模型移回 CPU 以节省内存
        model_l = model_l.cpu()
        
        return model_l

    def clients_update(self):
        """
        客户端更新：收集所有候选客户端的模型更新
        
        遍历所有候选客户端，使用 SAM 优化器进行本地训练，然后计算模型更新量。
        
        与 FedAvgM 保持一致，计算的是模型更新量（delta_model），而不是完整的模型。
        """
        self.delta_models = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            # 使用 SAM 优化器进行本地训练
            model = self.local_update(self.clients[k], self.server.model)
            # 计算模型更新量（本地模型 - 全局模型）
            delta_model = self.diff_model(model, self.server.model)
            self.delta_models.append(delta_model)

    def server_update(self):
        """
        服务器更新：聚合客户端更新并使用动量机制更新全局模型
        
        使用 FedAvgM 的聚合方式，通过动量机制平滑模型更新。
        
        更新公式：
        =========
        m_t = beta1 * m_{t-1} + (1 - beta1) * delta_w_t
        w_{t+1} = w_t - lr_g * m_t
        
        其中：
        - m_t: 动量缓冲区
        - delta_w_t: 聚合后的客户端更新
        - w_t: 当前全局模型参数
        - lr_g: 全局学习率
        - beta1: 动量系数
        """
        # 获取全局学习率和动量系数
        lr_g = self.args.lr_g
        beta1 = self.args.beta1
        
        # 获取当前全局模型参数
        w_t = self.server.model.state_dict()
        
        # 聚合客户端更新
        delta_w_t = self.averaging(self.delta_models)
        
        # 使用动量机制更新全局模型
        w = {}
        for key in w_t.keys():
            # 更新动量缓冲区：m_t = beta1 * m_{t-1} + (1 - beta1) * delta_w_t
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * delta_w_t[key]
            # 更新全局模型：w_{t+1} = w_t - lr_g * m_t
            w[key] = w_t[key] - lr_g * self.m[key]
        
        # 加载更新后的模型参数
        self.server.model.load_state_dict(w)

    def diff_model(self, model_l, model_g):
        """
        计算模型差异：本地模型与全局模型的差值
        
        用于计算客户端本地训练后的模型更新量。
        
        Args:
            model_l: 本地模型
            model_g: 全局模型
        
        Returns:
            delta_model: 模型更新量（本地模型 - 全局模型）
        """
        # 将模型移到 CPU 以节省内存
        model_l.cpu()
        
        # 获取模型参数字典
        w_l = model_l.state_dict()
        w_g = model_g.state_dict()
        
        # 计算参数差值：delta = w_l - w_g
        for key in w_l.keys():
            w_l[key] *= -1  # w_l = -w_l
            w_l[key] += w_g[key]  # w_l = w_g - w_l = delta
        
        return model_l

    def zero_weights(self, model):
        """
        将模型的所有参数初始化为零
        
        用于初始化动量缓冲区。
        
        Args:
            model: 要初始化的模型
        """
        for n, p in model.named_parameters():
            p.data.zero_()
