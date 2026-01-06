from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ==================== 常量定义 ====================

# L2O 网络架构配置
DEFAULT_INPUT_SIZE = 2
DEFAULT_HIDDEN_SIZE = 30
DEFAULT_OUTPUT_SIZE = 1
DEFAULT_NUM_LAYERS = 2
DEFAULT_TRAINING_INTERVAL = 3  # 聚合器训练间隔
DEFAULT_META_LEARNING_RATE = 0.001  # 元优化器学习率
DEFAULT_AGGREGATOR_TRAINING_EPOCHS = 5  # 聚合器训练轮数
DEFAULT_GRADIENT_SCALE_PARAM = 10  # 梯度预处理缩放参数


# ==================== FedL2O 主类定义 ====================

class FedL2O(FedAvgM):
    """
    FedL2O: 基于学习优化的联邦学习算法
    
    使用 L2O (Learning to Optimize) 模块进行智能的梯度聚合和模型更新。
    该方法通过学习一个优化器网络来自动学习最优的聚合和更新策略。
    
    Attributes:
        l2o_optimizer: L2O 优化器网络，用于学习聚合和更新策略
        meta_optimizer: 元优化器，用于训练 L2O 网络
        hidden_states: LSTM 隐藏状态列表
        truncated_hidden_states: 截断的隐藏状态（用于训练）
        training_mode: 是否处于训练模式
        training_step: 当前训练步数
        training_interval: 聚合器训练间隔
        model_checkpoints: 模型检查点列表
        gradient_checkpoints: 梯度检查点列表
    """

    def __init__(self, server, clients, args):
        """
        初始化 FedL2O
        
        Args:
            server: 服务器对象
            clients: 客户端列表
            args: 配置参数对象
        """
        super().__init__(server, clients, args)
        # 根据配置决定是否训练聚合器
        self.training_mode = not args.FL_validate_clients

    def server_init(self):
        """
        服务器初始化：初始化模型、代理数据和 L2O 优化器网络
        """
        # 初始化全局模型
        self.server.model = self.server.init_model()
        # 初始化代理数据（用于训练 L2O 网络）
        self.server.proxy_data = self.server.init_proxy_data()
        
        # L2O 网络架构参数
        input_size = DEFAULT_INPUT_SIZE
        hidden_size = DEFAULT_HIDDEN_SIZE
        output_size = DEFAULT_OUTPUT_SIZE
        num_layers = DEFAULT_NUM_LAYERS
        
        # 初始化 L2O 优化器网络
        # num_layers 包括输入层和隐层，但还需要接入输出层
        self.l2o_optimizer = L2O(
            input_size, hidden_size, output_size, num_layers, self.args
        ).to(self.args.device)
        
        # 如果不需要训练，则加载预训练的优化器网络
        if not self.training_mode:
            model_path = "./assets/" + "_".join(
                ["aggr", self.args.dataset, self.args.iid, str(self.args.seed)]
            ) + ".pt"
            self.l2o_optimizer = torch.load(model_path, map_location="cpu").to(self.args.device)
        
        # 定义优化器类型工厂
        optimizer_factory = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.AdamW
        }
        
        # 初始化元优化器（用于训练 L2O 网络）
        self.meta_optimizer = optimizer_factory[self.args.server_optimizer](
            self.l2o_optimizer.parameters(),
            lr=DEFAULT_META_LEARNING_RATE,
        )
        
        # 初始化训练相关变量
        self.training_step = 0
        self.training_interval = DEFAULT_TRAINING_INTERVAL
        self.model_checkpoints = []
        self.gradient_checkpoints = []
        
        # 初始化 LSTM 隐藏状态
        self._initialize_hidden_states(hidden_size, num_layers)

    def _initialize_hidden_states(self, hidden_size: int, num_layers: int):
        """
        初始化 LSTM 隐藏状态
        
        隐藏状态结构说明：
        =================
        hidden_states 是一个列表，包含 num_layers + 1 个元素：
        
        1. hidden_states[0:num_layers-1]: 隐藏层 LSTM 的状态
           - 每个元素: [h, c] - 标准 LSTM 的隐藏状态和单元状态
           - 形状: (2, hidden_size) 或 None（初始化为 None）
           - 作用: 用于提取梯度的高级特征表示
        
        2. hidden_states[-1]: 输出层 LSTM 的状态
           - 形状: (3, param_size)
           - 内容: [weight_decay, learning_rate, model_parameters]
             * [0]: weight_decay - 权重衰减状态（每个参数的权重衰减系数）
             * [1]: learning_rate - 学习率状态（每个参数的学习率）
             * [2]: model_parameters - 模型参数状态（当前模型参数的向量表示）
           - 作用: 存储当前模型参数以及为每个参数学习到的优化超参数
        
        初始化策略：
        ===========
        - 隐藏层状态: 初始化为 None，在第一次前向传播时自动初始化为零
        - 输出层状态: 
          * weight_decay: 初始化为零向量
          * learning_rate: 初始化为零向量
          * model_parameters: 初始化为当前服务器模型的参数向量
        
        Args:
            hidden_size: 隐藏层大小
            num_layers: LSTM 层数（不包括输出层）
        """
        # 初始化隐藏层状态（前 num_layers 层）
        # 这些状态用于隐藏层 LSTM，存储 [h, c] 状态
        self.hidden_states = [None] * num_layers
        
        # 获取模型参数向量（当前服务器模型的参数）
        # 这是输出层 LSTM 状态中模型参数部分的初始值
        model_parameters = self.server.model.to_vector().data.to(self.args.device)
        
        # 初始化权重衰减和学习率（用于输出层 LSTM）
        # 这些是输出层 LSTM 状态中的优化超参数部分
        # 初始化为零，LSTM 会在训练过程中学习这些值
        weight_decay = torch.zeros_like(model_parameters)  # 权重衰减状态
        learning_rate = torch.zeros_like(model_parameters)  # 学习率状态
        
        # 将权重衰减、学习率和模型参数堆叠作为最后一个隐藏状态
        # 输出层 LSTM 的状态格式：[weight_decay, learning_rate, model_parameters]
        # 形状: (3, param_size)
        # - [0]: 权重衰减状态（f_t）
        # - [1]: 学习率状态（i_t）
        # - [2]: 模型参数状态（c_t）- 这是当前模型参数的向量表示
        output_state = torch.stack([weight_decay, learning_rate, model_parameters], dim=0)
        self.hidden_states.append(output_state)
        
        # 初始化截断的隐藏状态（用于训练时重置）
        # 在训练 L2O 网络时，使用截断的隐藏状态可以避免梯度爆炸
        self.truncated_hidden_states = copy.deepcopy(self.hidden_states)

    def server_update(self):
        """
        服务器更新：聚合客户端梯度并更新全局模型
        
        该方法执行以下步骤：
        1. 收集客户端模型更新（梯度）
        2. 如果处于训练模式，保存检查点并定期训练 L2O 网络
        3. 使用 L2O 网络聚合梯度并更新全局模型
        4. 记录学习到的优化参数（学习率和权重衰减）
        """
        # 更新训练步数
        self.training_step += 1
        
        # 获取当前模型参数
        model_parameters = self.server.model.to_vector().data.to(self.args.device)
        
        # 收集客户端梯度
        client_gradients = self._collect_client_gradients()
        
        # 训练 L2O 优化器网络（如果处于训练模式）
        if self.training_mode:
            self._save_training_checkpoints(model_parameters, client_gradients)
            if self.training_step % self.training_interval == 0:
                self._train_optimizer_network()
                self._clear_training_checkpoints()
        
        # 使用 L2O 网络聚合梯度并更新模型
        with torch.set_grad_enabled(self.training_mode):
            # L2O 网络处理流程：
            # 1. 输入: client_gradients - 各客户端的模型梯度（参数更新量）
            # 2. L2A 聚合: 将多个客户端梯度聚合并预处理为特征 (param_size, 2)
            # 3. L2U 更新: LSTM 基于梯度特征和当前参数状态，学习新的参数值
            # 4. 输出: update_vector - LSTM 学习到的新模型参数向量
            #          updated_hidden_states - 更新后的 LSTM 状态
            #            - 包含学习率状态 (i_t)、权重衰减状态 (f_t)、模型参数状态 (c_t)
            update_vector, updated_hidden_states = self.l2o_optimizer(
                client_gradients, self.hidden_states
            )
            # 更新隐藏状态（分离计算图以避免梯度累积）
            # hidden_states[-1] 包含 [weight_decay, learning_rate, model_parameters]
            self.hidden_states = [state.data for state in updated_hidden_states]
            # 直接使用 LSTM 输出的新参数向量（与 fedleo 保持一致）
            # update_vector 来自 LSTM 输出层状态的 c_t（模型参数状态）
            # 注意: 这里直接替换模型参数，而不是使用梯度更新
            # θ_new = LSTM_output（直接替换，而非 θ_old - lr * gradient）
            self.server.model.from_vector(update_vector.data.cpu())
        
        # 记录学习到的优化参数
        self._record_learned_parameters()

    def _collect_client_gradients(self) -> torch.Tensor:
        """
        收集客户端梯度
        
        Returns:
            client_gradients: 客户端梯度张量，形状为 (param_size, num_clients)
        """
        client_gradients = [
            delta_model.to_vector() for delta_model in self.delta_models
        ]
        # 转置以便后续处理：从 (num_clients, param_size) 转为 (param_size, num_clients)
        client_gradients = torch.stack(client_gradients).data.T.to(self.args.device)
        return client_gradients

    def _save_training_checkpoints(self, model_parameters: torch.Tensor, 
                                   client_gradients: torch.Tensor):
        """
        保存训练检查点
        
        Args:
            model_parameters: 当前模型参数
            client_gradients: 客户端梯度
        """
        self.model_checkpoints.append(model_parameters.cpu())
        self.gradient_checkpoints.append(client_gradients.cpu())

    def _clear_training_checkpoints(self):
        """清空训练检查点"""
        self.model_checkpoints = []
        self.gradient_checkpoints = []

    def _record_learned_parameters(self):
        """
        记录学习到的优化参数（学习率和权重衰减）
        
        这些参数是从 L2O 网络的输出层 LSTM 状态中提取的。
        
        参数提取说明：
        =============
        - hidden_states[-1]: 输出层 LSTM 的状态，形状为 (3, param_size)
          * [0]: weight_decay_state - 权重衰减状态（每个参数的权重衰减系数）
          * [1]: learning_rate_state - 学习率状态（每个参数的学习率）
          * [2]: model_parameters - 模型参数状态（当前模型参数向量）
        
        - 这些状态是 LSTM 根据客户端梯度学习到的优化超参数
        - 与传统优化器不同，这里每个参数都有独立的学习率和权重衰减
        - 使用平方均值作为统计量，反映学习到的优化参数的总体水平
        """
        try:
            # 从输出层状态中提取学习率和权重衰减
            # hidden_states[-1] 格式：[weight_decay, learning_rate, model_parameters]
            # 形状: (3, param_size)
            output_state = self.hidden_states[-1]
            weight_decay_state = output_state[0].data  # 权重衰减状态 (param_size,)
            learning_rate_state = output_state[1].data  # 学习率状态 (param_size,)
            
            # 计算学习到的参数（使用平方均值作为统计量）
            # 这些值反映了 LSTM 为所有参数学习到的优化超参数的平均水平
            learned_learning_rate = torch.mean(learning_rate_state ** 2).item()
            learned_weight_decay = torch.mean(weight_decay_state ** 2).item()
            
            # 记录到记录器
            self.recorder({
                'learned_lr_g': learned_learning_rate,  # 学习到的学习率（全局统计）
                'learned_w_decay': learned_weight_decay  # 学习到的权重衰减（全局统计）
            })
        except Exception:
            # 如果记录失败，静默忽略（可能在某些初始化阶段状态尚未准备好）
            pass

    def _train_optimizer_network(self):
        """
        训练 L2O 优化器网络
        
        使用代理数据和保存的检查点来训练 L2O 网络，使其学习更好的聚合和更新策略。
        """
        # 创建代理模型（用于模拟训练过程）
        surrogate_model = copy.deepcopy(self.server.model)
        loss_function = surrogate_model.loss_fn
        
        # 创建代理数据集和数据加载器
        surrogate_dataset = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            surrogate_dataset, 
            batch_size=self.args.server_batch_size, 
            shuffle=True
        )
        
        # 训练多个 epoch
        for epoch in range(DEFAULT_AGGREGATOR_TRAINING_EPOCHS):
            for batch in train_loader:
                total_loss = 0.0
                batch = to_device(batch, self.args.device)
                
                # 使用截断的隐藏状态（避免梯度爆炸）
                hidden_states = copy.deepcopy(self.truncated_hidden_states)
                
                # 遍历所有梯度检查点，模拟多步更新
                for client_gradients in self.gradient_checkpoints:
                    client_gradients = client_gradients.to(self.args.device)
                    
                    # 使用 L2O 网络更新参数
                    updated_params, hidden_states = self.l2o_optimizer(
                        client_gradients, hidden_states
                    )
                    
                    # 将更新后的参数应用到代理模型
                    self._set_model_parameters_from_vector(surrogate_model, updated_params)
                    
                    # 计算预测损失
                    predictions = surrogate_model(batch)
                    labels = batch["label"]
                    prediction_loss = loss_function(predictions, labels)
                    
                    # 正则化损失（当前未使用，但可以添加）
                    regularization_loss = 0.0
                    # regularization_loss = torch.sum((model_parameters - updated_params) ** 2)
                    
                    total_loss += prediction_loss + regularization_loss
                
                # 计算平均损失
                total_loss /= len(self.model_checkpoints)
                
                # 反向传播和优化
                total_loss.backward()
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                
                # 打印训练进度
                print(f"Epoch {epoch}, Loss: {total_loss.data.item():.6f}")
        
        # 应用最后一次更新的参数到服务器模型
        self.server.model.from_vector(updated_params.cpu())
        # 更新隐藏状态
        self.hidden_states = [state.data for state in hidden_states]
        self.truncated_hidden_states = [state.data for state in hidden_states]

    def eval_proxy_loss(self, surrogate_model, proxy_data):
        """
        评估代理损失：在代理数据上评估模型性能
        
        Args:
            surrogate_model: 代理模型
            proxy_data: 代理数据
        
        Returns:
            average_loss: 平均损失值
        """
        total_loss = 0.0
        loss_function = surrogate_model.loss_fn
        
        # 创建代理数据集和数据加载器
        surrogate_dataset = Dataset(proxy_data, self.args)
        train_loader = DataLoader(
            surrogate_dataset, 
            batch_size=self.args.server_batch_size, 
            shuffle=True
        )
        
        # 训练多个 epoch
        for epoch in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader, start=1):
                batch = to_device(batch, self.args.device)
                predictions = surrogate_model(batch)
                labels = batch["label"]
                # 使用移动平均计算损失
                total_loss += (loss_function(predictions, labels) - total_loss) / idx
                print(f"Epoch {epoch}, Loss: {total_loss.data.item():.6f}")
        
        return total_loss

    def _set_model_parameters_from_vector(self, model, param_vector: torch.Tensor):
        """
        从参数向量设置模型参数
        
        将展平的参数向量按照模型结构分配到各个参数层。
        
        Args:
            model: 目标模型
            param_vector: 参数向量（展平的一维张量）
        """
        pointer = 0
        # 遍历模型的所有命名参数
        for parameter_name, parameter_tensor in model.named_parameters():
            num_parameters = parameter_tensor.numel()
            # 分割参数名称以获取模块路径
            split_name = parameter_name.split(".")
            
            # 导航到对应的模块
            module = model
            for module_name in split_name[:-1]:
                module = module._modules[module_name]
            
            # 从向量中提取参数并重塑形状
            parameter_value = param_vector[pointer:pointer + num_parameters].view_as(parameter_tensor)
            # 设置模块参数
            module._parameters[split_name[-1]] = parameter_value
            pointer += num_parameters

    def proxy_model_from_vector(self, model, vec):
        """
        从向量恢复模型参数（保持向后兼容的接口）
        
        Args:
            model: 目标模型
            vec: 参数向量
        """
        self._set_model_parameters_from_vector(model, vec)

    def load_aggregator(self, path: str):
        """
        加载聚合器：从文件加载预训练的 L2O 优化器网络
        
        Args:
            path: 模型文件路径
        """
        self.l2o_optimizer = torch.load(path)
        self.training_mode = False

    def train(self):
        """设置为训练模式"""
        self.training_mode = True

    def test(self):
        """设置为测试模式"""
        self.training_mode = False


# ==================== L2O 相关类定义 ====================

class L2O(nn.Module):
    """
    L2O (Learning to Optimize) - 学习优化模块
    
    整合 L2A（聚合）和 L2U（更新）两个部分，实现端到端的学习优化。
    基于 CoordinateLSTMOptimizer 修改而来，只是改变了结构组织。
    
    LSTM 输入输出对应关系总览：
    ===========================
    输入流程：
    1. client_gradients (param_size, num_clients)
       → 各客户端的模型梯度（参数更新量）
    
    2. L2A 聚合 → processed_input (param_size, 2)
       → 预处理后的梯度特征（幅值和符号）
    
    3. L2U 处理 → LSTM 网络
       → 隐藏层 LSTM: 提取梯度特征
       → 输出层 LSTM: 学习新的模型参数
    
    输出流程：
    1. updated_params (param_size,)
       → LSTM 学习到的新模型参数向量
       → 来源: hidden_states[-1][2] (输出层 LSTM 的 c_t)
    
    2. updated_hidden_states
       → hidden_states[0:num_layers-1]: 隐藏层 LSTM 的 [h, c] 状态
       → hidden_states[-1]: 输出层 LSTM 的 [i, f, c] 状态
         * i: 学习率状态（每个参数的学习率）
         * f: 权重衰减状态（每个参数的权重衰减系数）
         * c: 模型参数状态（新的模型参数向量）
    
    关键特点：
    =========
    - LSTM 直接输出新的模型参数值，而不是参数更新量
    - 每个参数都有独立的学习率和权重衰减（Coordinate Descent 风格）
    - 优化策略是通过学习得到的，而非人工设计
    
    Attributes:
        l2a: L2A 模块，负责聚合客户端梯度
        l2u: L2U 模块，负责模型参数更新
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int, args):
        """
        初始化 L2O 模块
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_layers: LSTM 层数
            args: 配置参数
        """
        super(L2O, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 为了与 CoordinateLSTMOptimizer 保持完全一致的参数初始化顺序
        # 必须按照相同的顺序定义模块
        self.input = nn.Linear(2, hidden_size)
        self.mh_attention = MultiheadAttention(
            hidden_size, num_heads=1, batch_first=True, bias=False
        )
        
        # LSTM 层：必须在这里定义，以保持与 CoordinateLSTMOptimizer 的参数初始化顺序一致
        self.lstms = nn.ModuleList([
            LSTMCell(
                input_size if i == 0 else hidden_size, 
                hidden_size, 
                bias=True
            )
            for i in range(num_layers)
        ])
        self.lstms.append(LSTMCell_out(hidden_size, bias=True))
        
        # 创建 L2A 和 L2U 模块，让它们引用 L2O 中的模块
        # 这样可以保持参数初始化顺序一致，同时通过 L2A 和 L2U 实现功能
        self.l2a = L2A(input_size, hidden_size, args, self.input, self.mh_attention)
        self.l2u = L2U(input_size, hidden_size, output_size, num_layers, args, self.lstms)

    def forward(self, client_gradients: torch.Tensor, 
                hidden_states: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播：先聚合梯度，再更新模型参数
        
        数据流说明：
        ===========
        1. 输入: client_gradients
           - 形状: (param_size, num_clients)
           - 内容: 各个客户端的模型梯度（delta_models 的向量化表示）
           - 来源: 客户端本地训练后的模型更新量（模型参数的变化量）
        
        2. 梯度聚合: 通过 L2A 模块聚合多个客户端的梯度
           - 平均聚合: 计算所有客户端梯度的平均值
           - 预处理: 将梯度转换为两个特征（幅值和符号）
           - 输出: processed_input，形状为 (param_size, 2)
        
        3. 参数更新: 通过 L2U 模块使用 LSTM 学习新的模型参数
           - 输入: processed_input（预处理后的聚合梯度）
           - LSTM 处理: 
             * 隐藏层 LSTM: 提取梯度特征
             * 输出层 LSTM: 基于梯度特征和当前参数状态，学习新的参数值
           - 输出: updated_params（新的模型参数向量）
        
        4. 隐藏状态: hidden_states 存储了 LSTM 的内部状态
           - 隐藏层状态: [h, c] - 用于特征提取
           - 输出层状态: [i, f, c] - 其中 c 是模型参数，i 是学习率，f 是权重衰减
        
        Args:
            client_gradients: 客户端梯度张量，形状为 (param_size, num_clients)
              - 每一列代表一个客户端的模型梯度（参数更新量）
            hidden_states: LSTM 的隐藏状态列表
              - 前 num_layers 个元素: 隐藏层 LSTM 的 [h, c] 状态
              - 最后一个元素: 输出层 LSTM 的 [i, f, c] 状态
                * i: 学习率状态（每个参数的学习率）
                * f: 权重衰减状态（每个参数的权重衰减系数）
                * c: 模型参数状态（当前模型参数的向量表示）
        
        Returns:
            updated_params: 更新后的模型参数向量，形状为 (param_size,)
              - 这是 LSTM 学习到的新的模型参数，将直接替换当前模型参数
            updated_hidden_states: 更新后的隐藏状态列表
        """
        # 如果隐藏状态为空，初始化为零
        if hidden_states is None:
            hidden_states = [
                torch.zeros(2, self.hidden_size).to(client_gradients.device)
            ] * self.num_layers

        # 方法1：使用平均聚合（当前实现，与原始 CoordinateLSTMOptimizer 保持一致）
        # 通过 L2A 进行聚合（当前使用平均聚合，L2A 内部会处理）
        # client_gradients: (param_size, num_clients) - 各客户端的梯度
        # averaged_gradients: (param_size,) - 平均后的梯度
        averaged_gradients = torch.mean(client_gradients, dim=-1) / self.args.lr_l
        # processed_input: (param_size, 2) - 预处理后的梯度特征
        #   - [:, 0]: 梯度的幅值特征（对数缩放）
        #   - [:, 1]: 梯度的符号特征（指数缩放）
        processed_input = self.l2a.aggregate(averaged_gradients)
        
        # 方法2：使用 L2A 进行智能聚合（可选，当前被注释，与原始代码保持一致）
        # 如果需要使用注意力机制聚合，可以取消以下注释：
        # processed_input = self.l2a.aggregate_with_attention(client_gradients)
        
        # 使用 L2U 进行模型参数更新（与 CoordinateLSTMOptimizer 保持一致）
        # 输入: processed_input - 预处理后的聚合梯度特征
        # 输出: updated_params - LSTM 学习到的新模型参数向量
        #      hidden_states - 更新后的 LSTM 状态（包含学习率、权重衰减、模型参数）
        updated_params, hidden_states = self.l2u.forward(processed_input, hidden_states)
        
        return updated_params, hidden_states

    def preprocess_input(self, gradient, p=10):
        """
        输入预处理函数：将梯度转换为两个特征（幅值和符号）
        
        与 CoordinateLSTMOptimizer.preprocess_input 保持完全一致。
        
        Args:
            gradient: 梯度张量
            p: 缩放参数，默认值为 10
        
        Returns:
            processed_input: 处理后的输入，形状为 (..., 2)
        """
        # 裁剪幅值：使用对数缩放
        gradient_norm = torch.clamp(torch.log(torch.abs(gradient)) / p, min=-1)
        # 裁剪符号：使用指数缩放
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(p)) * gradient, min=-1, max=+1
        )
        # 堆叠幅值和符号特征
        processed_input = torch.stack([gradient_norm, gradient_sign], dim=-1)
        return processed_input


class L2A(nn.Module):
    """
    L2A (Learning to Aggregate) - 学习聚合模块
    
    负责对客户端梯度进行智能聚合，使用多头注意力机制学习最优的聚合权重。
    基于 CoordinateLSTMOptimizer 的聚合部分修改而来。
    
    Attributes:
        input_layer: 输入线性层，将预处理后的梯度映射到隐藏空间（引用自 L2O）
        mh_attention: 多头注意力机制，用于聚合多个客户端的梯度（引用自 L2O）
    """
    
    def __init__(self, input_size: int, hidden_size: int, args, input_layer, mh_attention):
        """
        初始化 L2A 模块
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            args: 配置参数
            input_layer: 输入线性层（从 L2O 传入，保持参数初始化顺序一致）
            mh_attention: 多头注意力机制（从 L2O 传入，保持参数初始化顺序一致）
        """
        super(L2A, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_layer = input_layer
        self.mh_attention = mh_attention

    def aggregate(self, averaged_gradients: torch.Tensor) -> torch.Tensor:
        """
        聚合方法：对平均梯度进行预处理（当前实现，与 CoordinateLSTMOptimizer 保持一致）
        
        Args:
            averaged_gradients: 平均后的梯度张量，形状为 (param_size,)
        
        Returns:
            processed_input: 处理后的输入，形状为 (param_size, 2)
        """
        # 预处理梯度输入（与 CoordinateLSTMOptimizer 保持一致）
        return self.preprocess_input(averaged_gradients)

    def aggregate_with_attention(self, client_gradients: torch.Tensor) -> torch.Tensor:
        """
        使用注意力机制聚合客户端梯度（可选方法，当前未使用）
        
        Args:
            client_gradients: 客户端梯度张量，形状为 (param_size, num_clients)
        
        Returns:
            aggregated_gradients: 聚合后的梯度表示
        """
        # 预处理输入梯度
        preprocessed = self.preprocess_input(client_gradients)
        # 通过线性层映射到隐藏空间
        hidden_representation = self.input_layer(preprocessed)
        # 使用多头注意力机制进行聚合
        aggregated_representation, _ = self.mh_attention(hidden_representation)
        # 返回聚合后的表示（取第一个元素，因为 batch_first=True）
        aggregated_gradients = aggregated_representation[0]
        return aggregated_gradients

    def preprocess_input(self, gradient: torch.Tensor, 
                         scale_param: float = DEFAULT_GRADIENT_SCALE_PARAM) -> torch.Tensor:
        """
        输入预处理函数：将梯度转换为两个特征（幅值和符号）
        
        使用对数缩放处理幅值，使用指数缩放处理符号，以便更好地处理不同尺度的梯度。
        
        Args:
            gradient: 梯度张量
            scale_param: 缩放参数，默认值为 10
        
        Returns:
            processed_input: 处理后的输入，形状为 (..., 2)
        """
        # 裁剪幅值：使用对数缩放
        gradient_norm = torch.clamp(
            torch.log(torch.abs(gradient)) / scale_param, min=-1
        )
        # 裁剪符号：使用指数缩放
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(scale_param)) * gradient, min=-1, max=+1
        )
        # 堆叠幅值和符号特征
        processed_input = torch.stack([gradient_norm, gradient_sign], dim=-1)
        return processed_input


class L2U(nn.Module):
    """
    L2U (Learning to Update) - 学习更新模块
    
    负责使用 LSTM 网络学习模型参数的更新策略，为每个参数学习个性化的
    学习率和权重衰减。基于 CoordinateLSTMOptimizer 的优化器部分修改而来。
    
    Attributes:
        lstms: LSTM 层列表，包括隐藏层和输出层（引用自 L2O）
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int, args, lstms):
        """
        初始化 L2U 模块
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_layers: LSTM 层数
            args: 配置参数
            lstms: LSTM 层列表（从 L2O 传入，保持参数初始化顺序一致）
        """
        super(L2U, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstms = lstms

    def forward(self, input_features: torch.Tensor, 
                hidden_states: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播：使用 LSTM 网络处理输入并更新模型参数
        
        LSTM 输入输出对应关系说明：
        ============================
        输入 (input_features):
        - 形状: (param_size, 2)
        - 内容: 聚合后的客户端梯度，经过预处理转换为两个特征
          * input_features[:, 0]: 梯度的幅值（对数缩放后的梯度绝对值）
          * input_features[:, 1]: 梯度的符号（指数缩放后的梯度符号）
        - 来源: 来自 L2A 模块聚合的客户端模型梯度（delta_models）
        
        隐藏层 LSTM (LSTMCell) 处理流程:
        - 输入: 预处理后的梯度特征 (param_size, 2) 或前一层 LSTM 的隐藏状态
        - 输出: [h_t, c_t] - 标准 LSTM 的隐藏状态和单元状态
        - 作用: 提取梯度的高级特征表示，为输出层 LSTM 提供特征输入
        
        输出层 LSTM (LSTMCell_out) 处理流程:
        - 输入: 最后一层隐藏层 LSTM 的隐藏状态 h_t (param_size, hidden_size)
        - 状态: [i_prev, f_prev, c_prev] - 前一时间步的状态
          * i_prev: 学习率状态（对应每个模型参数的学习率）
          * f_prev: 权重衰减状态（对应每个模型参数的权重衰减系数）
          * c_prev: 模型参数状态（当前模型参数的向量表示）
        - 输出: [i_t, f_t, c_t] - 更新后的状态
          * i_t: 更新后的学习率（用于参数更新的步长）
          * f_t: 更新后的权重衰减（用于正则化）
          * c_t: 更新后的模型参数（直接作为新的模型参数向量）
        
        最终输出 (updated_params):
        - 来源: hidden_states[-1][-1]，即输出层 LSTM 的 c_t（模型参数状态）
        - 形状: (param_size,)
        - 含义: 这是 LSTM 学习到的新的模型参数向量，将直接替换当前模型参数
        - 注意: 与传统的梯度下降不同，这里 LSTM 直接输出新的参数值，而不是参数更新量
        
        Args:
            input_features: 输入特征（通常是聚合后的梯度），形状为 (param_size, 2)
            hidden_states: LSTM 的隐藏状态列表
              - hidden_states[0:num_layers-1]: 隐藏层 LSTM 的 [h, c] 状态
              - hidden_states[-1]: 输出层 LSTM 的 [i, f, c] 状态，其中：
                * [0]: 权重衰减 (weight_decay)
                * [1]: 学习率 (learning_rate)  
                * [2]: 模型参数 (model_parameters)
        
        Returns:
            updated_params: 更新后的模型参数向量，形状为 (param_size,)
            updated_hidden_states: 更新后的隐藏状态列表
        """
        # 如果隐藏状态为空，初始化为零
        if hidden_states is None:
            hidden_states = [
                torch.zeros(2, self.hidden_size).to(input_features.device)
            ] * self.num_layers

        # 通过每一层 LSTM 进行处理（与 CoordinateLSTMOptimizer 保持一致）
        # 输入: input_features 是预处理后的梯度特征 (param_size, 2)
        #       - 第0维: 梯度的幅值特征
        #       - 第1维: 梯度的符号特征
        current_input = input_features
        for layer_idx, lstm_layer in enumerate(self.lstms):
            # 隐藏层 LSTM: 处理梯度特征，提取高级表示
            # 输出层 LSTM: 基于梯度特征和当前参数状态，学习新的参数值
            hidden_states[layer_idx] = lstm_layer(current_input, hidden_states[layer_idx])
            # 使用隐藏状态 h_t 作为下一层的输入
            current_input = hidden_states[layer_idx][0]

        # hidden_states[-1] 包含了输出层 LSTM 的状态 [weight_decay, learning_rate, model_parameters]
        # hidden_states[-1][0]: 权重衰减状态（f_t）
        # hidden_states[-1][1]: 学习率状态（i_t）
        # hidden_states[-1][2]: 模型参数状态（c_t）- 这是最终输出的新参数向量
        # hidden_states[:-1] 是前 num_layers 层 LSTM 的 [h, c] 状态
        updated_params = hidden_states[-1][-1]  # 提取模型参数（c_t）
        return updated_params, hidden_states

    def preprocess_input(self, gradient: torch.Tensor, 
                        scale_param: float = DEFAULT_GRADIENT_SCALE_PARAM) -> torch.Tensor:
        """
        输入预处理函数：将梯度转换为两个特征（幅值和符号）
        
        Args:
            gradient: 梯度张量
            scale_param: 缩放参数，默认值为 10
        
        Returns:
            processed_input: 处理后的输入，形状为 (..., 2)
        """
        # 裁剪幅值：使用对数缩放
        gradient_norm = torch.clamp(
            torch.log(torch.abs(gradient)) / scale_param, min=-1
        )
        # 裁剪符号：使用指数缩放
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(scale_param)) * gradient, min=-1, max=+1
        )
        # 堆叠幅值和符号特征
        processed_input = torch.stack([gradient_norm, gradient_sign], dim=-1)
        return processed_input


# ==================== LSTM 相关类定义 ====================

class LSTMCell(nn.Module):
    """
    自定义 LSTM 单元模块（隐藏层）
    
    用于 L2O 中的隐藏层处理，实现标准的 LSTM 功能。
    
    为什么有两种 LSTM？
    ====================
    1. LSTMCell（隐藏层 LSTM）：
       - 用于中间层的特征提取和转换
       - 输入和隐藏状态都是向量（hidden_size 维度）
       - 返回标准的 [h_t, c_t]（隐藏状态和单元状态）
       - 用于处理序列信息，提取高级特征
    
    2. LSTMCell_out（输出层 LSTM）：
       - 用于输出层，直接控制模型参数的更新
       - 输入是 hidden_size 维度的向量
       - 但状态是 3 个标量：i（学习率）、f（权重衰减）、c（模型参数）
       - 返回 [i_t, f_t, c_t]，用于为每个模型参数学习个性化的学习率和权重衰减
       - 这是 Coordinate Descent 风格的设计，允许为每个参数独立学习优化策略
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = False):
        """
        初始化 LSTM 单元
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            bias: 是否使用偏置
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        
        # 输出门参数
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化函数，使用 Xavier 初始化"""
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for param in self.parameters():
            torch.nn.init.uniform_(param, -std, std)

    def forward(self, inputs: torch.Tensor, 
                state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            inputs: 输入张量
            state: 前一个时间步的状态 (h_prev, c_prev)，如果为 None 则初始化为零
        
        Returns:
            stacked_state: 堆叠的状态 [h_t, c_t]
        """
        if state is None:
            h_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            c_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            state = (h_t, c_t)

        h_prev, c_prev = state
        
        # 输入门
        i_t = torch.sigmoid(inputs @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
        # 遗忘门
        f_t = torch.sigmoid(inputs @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
        # 输出门
        o_t = torch.sigmoid(inputs @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)
        # 单元状态候选值
        g_t = torch.tanh(inputs @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
        
        # 更新单元状态
        c_t = f_t * c_prev + i_t * g_t
        # 更新隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return torch.stack([h_t, c_t], dim=0)


class LSTMCell_out(nn.Module):
    """
    自定义 LSTM 单元模块（输出层）
    
    用于 L2O 中的输出层，控制学习率和权重衰减。
    这个 LSTM 的设计与标准 LSTM 不同：
    - 状态是 3 个标量：[i, f, c]，分别代表学习率、权重衰减和模型参数
    - 用于为每个模型参数学习个性化的优化策略（Coordinate Descent 风格）
    - 这是为什么需要两种 LSTM 的原因：输出层需要直接控制参数更新，而不是提取特征
    """
    
    def __init__(self, hidden_size: int, bias: bool = True):
        """
        初始化输出层 LSTM 单元
        
        Args:
            hidden_size: 隐藏层维度
            bias: 是否使用偏置
        """
        super(LSTMCell_out, self).__init__()
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hi = nn.Parameter(torch.Tensor(3, 1))
        self.b_i = nn.Parameter(torch.Tensor(1)) if bias else 0
        
        # 遗忘门参数
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hf = nn.Parameter(torch.Tensor(3, 1))
        self.b_f = nn.Parameter(torch.Tensor(1)) if bias else 0
        
        # 单元状态参数
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hg = nn.Parameter(torch.Tensor(3, 1))
        self.b_g = nn.Parameter(torch.Tensor(1)) if bias else 0
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化函数，使用 Xavier 初始化"""
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for param in self.parameters():
            torch.nn.init.uniform_(param, -std, std)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        输出层 LSTM 的输入输出对应关系：
        ================================
        输入 (inputs):
        - 形状: (param_size, hidden_size)
        - 内容: 最后一层隐藏层 LSTM 的隐藏状态 h_t
        - 含义: 基于客户端梯度提取的高级特征表示
        
        状态 (state):
        - 形状: (3, param_size)
        - 内容: [i_prev, f_prev, c_prev]
          * i_prev: 前一时间步的学习率状态（每个参数的学习率）
          * f_prev: 前一时间步的权重衰减状态（每个参数的权重衰减系数）
          * c_prev: 前一时间步的模型参数状态（当前模型参数的向量表示）
        
        输出 (stacked_state):
        - 形状: (3, param_size)
        - 内容: [i_t, f_t, c_t]
          * i_t: 更新后的学习率状态
            - 对应模型参数更新的步长
            - 用于控制参数更新的幅度
          * f_t: 更新后的权重衰减状态
            - 对应模型参数的正则化系数
            - 用于防止过拟合
          * c_t: 更新后的模型参数状态
            - 这是最终输出的新模型参数向量
            - 计算公式: c_t = (1 - f_t) * c_prev + i_t * g_t
            - 其中 g_t 是基于当前梯度特征和参数状态学习到的参数更新量
            - 注意: 这里直接输出新的参数值，而不是参数更新量（delta）
        
        参数更新机制：
        =============
        - 与传统优化器（如 SGD）不同，这里 LSTM 直接学习新的参数值
        - 传统方式: θ_new = θ_old - lr * gradient
        - L2O 方式: θ_new = LSTM(gradient_features, [lr_state, wd_state, θ_old])
        - LSTM 会根据梯度特征和历史状态，为每个参数学习个性化的更新策略
        
        Args:
            inputs: 输入张量，形状为 (param_size, hidden_size)
              - 来自最后一层隐藏层 LSTM 的隐藏状态 h_t
              - 包含基于客户端梯度提取的特征信息
            state: 前一个时间步的状态，形状为 (3, param_size)
              - state[0]: i_prev - 学习率状态
              - state[1]: f_prev - 权重衰减状态
              - state[2]: c_prev - 模型参数状态（当前模型参数向量）
        
        Returns:
            stacked_state: 堆叠的状态，形状为 (3, param_size)
              - [0]: i_t - 更新后的学习率状态
              - [1]: f_t - 更新后的权重衰减状态
              - [2]: c_t - 更新后的模型参数状态（新的模型参数向量）
        """
        i_prev, f_prev, c_prev = state
        
        # 输入门: 控制新信息（基于梯度特征）的流入
        # i_t 对应学习率，控制参数更新的幅度
        input_gate_logit = inputs @ self.W_ii + state.T @ self.W_hi + self.b_i
        i_t = torch.tanh(input_gate_logit) * input_gate_logit
        
        # 遗忘门: 控制旧信息（当前参数）的保留
        # f_t 对应权重衰减，控制参数的正则化程度
        forget_gate_logit = inputs @ self.W_if + state.T @ self.W_hf + self.b_f
        f_t = torch.tanh(forget_gate_logit) * forget_gate_logit
        
        # 单元状态候选值: 基于梯度特征和当前状态学习参数更新量
        # g_t 是基于当前梯度特征学习到的参数更新候选值
        cell_gate_logit = inputs @ self.W_ig + state.T @ self.W_hg + self.b_g
        g_t = torch.tanh(cell_gate_logit)
        
        # 压缩维度
        i_t = i_t.squeeze(-1)  # 学习率状态 (param_size,)
        f_t = f_t.squeeze(-1)  # 权重衰减状态 (param_size,)
        g_t = g_t.squeeze(-1)  # 参数更新候选值 (param_size,)
        
        # 更新单元状态（模型参数）
        # c_t = (1 - f_t) * c_prev + i_t * g_t
        # - (1 - f_t) * c_prev: 保留部分旧参数（权重衰减控制）
        # - i_t * g_t: 添加新学习到的参数更新（学习率控制）
        # 最终 c_t 就是新的模型参数向量
        c_t = (1 - f_t) * c_prev + i_t * g_t

        return torch.stack([i_t, f_t, c_t], dim=0)


class MultiheadAttention(nn.Module):
    """
    多头注意力机制模块
    
    用于 L2A 聚合中的注意力计算，学习客户端梯度之间的重要性权重。
    """
    
    def __init__(self, hidden_size: int, num_heads: int, 
                 batch_first: bool = True, bias: bool = False):
        """
        初始化多头注意力模块
        
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            batch_first: 是否 batch 维度在前
            bias: 是否使用偏置
        """
        super(MultiheadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query, Key, Value 线性投影层（与 lstm.py 中的 MultiheadAttention 保持一致）
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 最终输出线性层（与 lstm.py 中的 MultiheadAttention 保持一致）
        self.fc = nn.Linear(hidden_size, hidden_size)

        # 缩放因子（用于缩放点积注意力）
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, hidden_size)
        
        Returns:
            output: 输出张量，形状为 (batch_size, seq_length, hidden_size)
            attention_weights: 注意力权重，形状为 (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, _ = x.size()

        # 线性投影得到 Q, K, V
        Q = self.query(x)  # (batch_size, seq_length, hidden_size)
        K = self.key(x)     # (batch_size, seq_length, hidden_size)
        V = self.value(x)   # (batch_size, seq_length, hidden_size)

        # 分割成多个头并转置
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V shape: (batch_size, num_heads, seq_length, head_dim)

        # 缩放点积注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # attention_scores shape: (batch_size, num_heads, seq_length, seq_length)

        attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_length, seq_length)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        # output shape: (batch_size, num_heads, seq_length, head_dim)

        # 拼接多个头并通过最终线性层
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        # output shape: (batch_size, seq_length, hidden_size)

        output = self.fc(output)
        # output shape: (batch_size, seq_length, hidden_size)

        return output, attention_weights
