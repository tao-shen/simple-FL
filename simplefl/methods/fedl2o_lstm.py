"""
FedL2O_LSTM：基于学习优化的联邦学习算法（LSTM 优化器，独立实现）。
与 fedl2o、fedl2o_mamba 为三种并列方法，本文件仅暴露 FedL2O_LSTM / L2O_LSTM。
"""
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

# L2O 网络架构配置（LSTM 版本）
DEFAULT_INPUT_SIZE = 2
DEFAULT_HIDDEN_SIZE = 30
DEFAULT_OUTPUT_SIZE = 1
DEFAULT_NUM_LAYERS = 2
DEFAULT_TRAINING_INTERVAL = 3  # 聚合器训练间隔
DEFAULT_META_LEARNING_RATE = 0.001  # 元优化器学习率
DEFAULT_AGGREGATOR_TRAINING_EPOCHS = 5  # 聚合器训练轮数
DEFAULT_GRADIENT_SCALE_PARAM = 10  # 梯度预处理缩放参数


# ==================== FedL2O_LSTM 主类定义 ====================

class FedL2O_LSTM(FedAvgM):
    """
    FedL2O (LSTM 版本): 基于学习优化的联邦学习算法，使用 LSTM 作为优化器网络。

    使用 L2O (Learning to Optimize) 模块进行智能的梯度聚合和模型更新。
    该方法通过学习一个 LSTM 优化器网络来自动学习最优的聚合和更新策略。

    Attributes:
        l2o_optimizer: L2O 优化器网络（LSTM），用于学习聚合和更新策略
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
        super().__init__(server, clients, args)
        self.training_mode = not args.FL_validate_clients

    def server_init(self):
        self.server.model = self.server.init_model()
        self.server.proxy_data = self.server.init_proxy_data()

        input_size = DEFAULT_INPUT_SIZE
        hidden_size = DEFAULT_HIDDEN_SIZE
        output_size = DEFAULT_OUTPUT_SIZE
        num_layers = DEFAULT_NUM_LAYERS

        self.l2o_optimizer = L2O_LSTM(
            input_size, hidden_size, output_size, num_layers, self.args
        ).to(self.args.device)

        if not self.training_mode:
            model_path = "./assets/" + "_".join(
                ["aggr", self.args.dataset, self.args.iid, str(self.args.seed)]
            ) + ".pt"
            self.l2o_optimizer = torch.load(model_path, map_location="cpu").to(self.args.device)

        optimizer_factory = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.AdamW
        }
        self.meta_optimizer = optimizer_factory[self.args.server_optimizer](
            self.l2o_optimizer.parameters(),
            lr=DEFAULT_META_LEARNING_RATE,
        )

        self.training_step = 0
        self.training_interval = DEFAULT_TRAINING_INTERVAL
        self.model_checkpoints = []
        self.gradient_checkpoints = []
        self.initialize_hidden_states(hidden_size, num_layers)

    def initialize_hidden_states(self, hidden_size: int, num_layers: int):
        self.hidden_states = [None] * num_layers
        model_parameters = self.server.model.to_vector().data.to(self.args.device)
        weight_decay = torch.zeros_like(model_parameters)
        learning_rate = torch.zeros_like(model_parameters)
        output_state = torch.stack([weight_decay, learning_rate, model_parameters], dim=0)
        self.hidden_states.append(output_state)
        self.truncated_hidden_states = copy.deepcopy(self.hidden_states)

    def server_update(self):
        self.training_step += 1
        model_parameters = self.server.model.to_vector().data.to(self.args.device)
        client_gradients = self._collect_client_gradients()

        if self.training_mode:
            self._save_training_checkpoints(model_parameters, client_gradients)
            if self.training_step % self.training_interval == 0:
                self._train_optimizer_network()
                self._clear_training_checkpoints()

        with torch.set_grad_enabled(self.training_mode):
            update_vector, updated_hidden_states = self.l2o_optimizer(
                client_gradients, self.hidden_states
            )
            self.hidden_states = [state.data for state in updated_hidden_states]
            self.server.model.from_vector(update_vector.data.cpu())

        self._record_learned_parameters()

    def _collect_client_gradients(self) -> torch.Tensor:
        client_gradients = [
            delta_model.to_vector() for delta_model in self.delta_models
        ]
        client_gradients = torch.stack(client_gradients).data.T.to(self.args.device)
        return client_gradients

    def _save_training_checkpoints(self, model_parameters: torch.Tensor,
                                   client_gradients: torch.Tensor):
        self.model_checkpoints.append(model_parameters.cpu())
        self.gradient_checkpoints.append(client_gradients.cpu())

    def _clear_training_checkpoints(self):
        self.model_checkpoints = []
        self.gradient_checkpoints = []

    def _record_learned_parameters(self):
        try:
            output_state = self.hidden_states[-1]
            weight_decay_state = output_state[0].data
            learning_rate_state = output_state[1].data
            learned_learning_rate = torch.mean(learning_rate_state ** 2).item()
            learned_weight_decay = torch.mean(weight_decay_state ** 2).item()
            self.recorder({
                'learned_lr_g': learned_learning_rate,
                'learned_w_decay': learned_weight_decay
            })
        except Exception:
            pass

    def _train_optimizer_network(self):
        surrogate_model = copy.deepcopy(self.server.model)
        loss_function = surrogate_model.loss_fn
        surrogate_dataset = Dataset(self.server.proxy_data, self.args)
        train_loader = DataLoader(
            surrogate_dataset,
            batch_size=self.args.server_batch_size,
            shuffle=True
        )

        for epoch in range(DEFAULT_AGGREGATOR_TRAINING_EPOCHS):
            for batch in train_loader:
                total_loss = 0.0
                batch = to_device(batch, self.args.device)
                hidden_states = copy.deepcopy(self.truncated_hidden_states)

                for client_gradients in self.gradient_checkpoints:
                    client_gradients = client_gradients.to(self.args.device)
                    updated_params, hidden_states = self.l2o_optimizer(
                        client_gradients, hidden_states
                    )
                    self._set_model_parameters_from_vector(surrogate_model, updated_params)
                    predictions = surrogate_model(batch)
                    labels = batch["label"]
                    prediction_loss = loss_function(predictions, labels)
                    regularization_loss = 0.0
                    total_loss += prediction_loss + regularization_loss

                total_loss /= len(self.model_checkpoints)
                total_loss.backward()
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                print(f"Epoch {epoch}, Loss: {total_loss.data.item():.6f}")

        self.server.model.from_vector(updated_params.cpu())
        self.hidden_states = [state.data for state in hidden_states]
        self.truncated_hidden_states = [state.data for state in hidden_states]

    def eval_proxy_loss(self, surrogate_model, proxy_data):
        total_loss = 0.0
        loss_function = surrogate_model.loss_fn
        surrogate_dataset = Dataset(proxy_data, self.args)
        train_loader = DataLoader(
            surrogate_dataset,
            batch_size=self.args.server_batch_size,
            shuffle=True
        )
        for epoch in range(self.args.server_epochs):
            for idx, batch in enumerate(train_loader, start=1):
                batch = to_device(batch, self.args.device)
                predictions = surrogate_model(batch)
                labels = batch["label"]
                total_loss += (loss_function(predictions, labels) - total_loss) / idx
                print(f"Epoch {epoch}, Loss: {total_loss.data.item():.6f}")
        return total_loss

    def _set_model_parameters_from_vector(self, model, param_vector: torch.Tensor):
        pointer = 0
        for parameter_name, parameter_tensor in model.named_parameters():
            num_parameters = parameter_tensor.numel()
            split_name = parameter_name.split(".")
            module = model
            for module_name in split_name[:-1]:
                module = module._modules[module_name]
            parameter_value = param_vector[pointer:pointer + num_parameters].view_as(parameter_tensor)
            module._parameters[split_name[-1]] = parameter_value
            pointer += num_parameters

    def proxy_model_from_vector(self, model, vec):
        self._set_model_parameters_from_vector(model, vec)

    def load_aggregator(self, path: str):
        self.l2o_optimizer = torch.load(path)
        self.training_mode = False

    def train(self):
        self.training_mode = True

    def test(self):
        self.training_mode = False


# ==================== L2O_LSTM 相关类定义 ====================

class L2O_LSTM(nn.Module):
    """
    L2O (LSTM 版本) - 学习优化模块，使用 LSTM 作为优化器网络。
    整合 L2A（聚合）和 L2U（更新），L2U 使用多层 LSTM 学习参数更新。
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int, args):
        super(L2O_LSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.input = nn.Linear(2, hidden_size)
        self.mh_attention = MultiheadAttention(
            hidden_size, num_heads=1, batch_first=True, bias=False
        )
        self.lstms = nn.ModuleList([
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=True
            )
            for i in range(num_layers)
        ])
        self.lstms.append(LSTMCell_out(hidden_size, bias=True))
        self.l2a = L2A(input_size, hidden_size, args, self.input, self.mh_attention)
        self.l2u = L2U(input_size, hidden_size, output_size, num_layers, args, self.lstms)

    def forward(self, client_gradients: torch.Tensor,
                hidden_states: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if hidden_states is None:
            hidden_states = [
                torch.zeros(2, self.hidden_size).to(client_gradients.device)
            ] * self.num_layers

        averaged_gradients = torch.mean(client_gradients, dim=-1) / self.args.lr_l
        processed_input = self.l2a.aggregate(averaged_gradients)
        updated_params, hidden_states = self.l2u.forward(processed_input, hidden_states)
        return updated_params, hidden_states

    def preprocess_input(self, gradient, p=10):
        gradient_norm = torch.clamp(torch.log(torch.abs(gradient)) / p, min=-1)
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(p)) * gradient, min=-1, max=+1
        )
        return torch.stack([gradient_norm, gradient_sign], dim=-1)


class L2A(nn.Module):
    """L2A (Learning to Aggregate) - 学习聚合模块。"""
    def __init__(self, input_size: int, hidden_size: int, args, input_layer, mh_attention):
        super(L2A, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_layer = input_layer
        self.mh_attention = mh_attention

    def aggregate(self, averaged_gradients: torch.Tensor) -> torch.Tensor:
        return self.preprocess_input(averaged_gradients)

    def aggregate_with_attention(self, client_gradients: torch.Tensor) -> torch.Tensor:
        preprocessed = self.preprocess_input(client_gradients)
        hidden_representation = self.input_layer(preprocessed)
        aggregated_representation, _ = self.mh_attention(hidden_representation)
        return aggregated_representation[0]

    def preprocess_input(self, gradient: torch.Tensor,
                         scale_param: float = DEFAULT_GRADIENT_SCALE_PARAM) -> torch.Tensor:
        gradient_norm = torch.clamp(
            torch.log(torch.abs(gradient)) / scale_param, min=-1
        )
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(scale_param)) * gradient, min=-1, max=+1
        )
        return torch.stack([gradient_norm, gradient_sign], dim=-1)


class L2U(nn.Module):
    """L2U (Learning to Update) - 使用 LSTM 学习参数更新。"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int, args, lstms):
        super(L2U, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstms = lstms

    def forward(self, input_features: torch.Tensor,
                hidden_states: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if hidden_states is None:
            hidden_states = [
                torch.zeros(2, self.hidden_size).to(input_features.device)
            ] * self.num_layers
        current_input = input_features
        for layer_idx, lstm_layer in enumerate(self.lstms):
            hidden_states[layer_idx] = lstm_layer(current_input, hidden_states[layer_idx])
            current_input = hidden_states[layer_idx][0]
        updated_params = hidden_states[-1][-1]
        return updated_params, hidden_states

    def preprocess_input(self, gradient: torch.Tensor,
                         scale_param: float = DEFAULT_GRADIENT_SCALE_PARAM) -> torch.Tensor:
        gradient_norm = torch.clamp(
            torch.log(torch.abs(gradient)) / scale_param, min=-1
        )
        gradient_sign = torch.clamp(
            torch.exp(torch.tensor(scale_param)) * gradient, min=-1, max=+1
        )
        return torch.stack([gradient_norm, gradient_sign], dim=-1)


class LSTMCell(nn.Module):
    """自定义 LSTM 单元（隐藏层）"""
    def __init__(self, input_size: int, hidden_size: int, bias: bool = False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size)) if bias else 0
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for param in self.parameters():
            torch.nn.init.uniform_(param, -std, std)

    def forward(self, inputs: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if state is None:
            h_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            c_t = torch.zeros(self.hidden_size, dtype=torch.float).to(inputs.device)
            state = (h_t, c_t)
        h_prev, c_prev = state
        i_t = torch.sigmoid(inputs @ self.W_ii.T + h_prev @ self.W_hi.T + self.b_i)
        f_t = torch.sigmoid(inputs @ self.W_if.T + h_prev @ self.W_hf.T + self.b_f)
        o_t = torch.sigmoid(inputs @ self.W_io.T + h_prev @ self.W_ho.T + self.b_o)
        g_t = torch.tanh(inputs @ self.W_ig.T + h_prev @ self.W_hg.T + self.b_g)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return torch.stack([h_t, c_t], dim=0)


class LSTMCell_out(nn.Module):
    """自定义 LSTM 单元（输出层），输出 [i, f, c] 状态"""
    def __init__(self, hidden_size: int, bias: bool = True):
        super(LSTMCell_out, self).__init__()
        self.hidden_size = hidden_size
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hi = nn.Parameter(torch.Tensor(3, 1))
        self.b_i = nn.Parameter(torch.Tensor(1)) if bias else 0
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hf = nn.Parameter(torch.Tensor(3, 1))
        self.b_f = nn.Parameter(torch.Tensor(1)) if bias else 0
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.W_hg = nn.Parameter(torch.Tensor(3, 1))
        self.b_g = nn.Parameter(torch.Tensor(1)) if bias else 0
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        for param in self.parameters():
            torch.nn.init.uniform_(param, -std, std)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        i_prev, f_prev, c_prev = state
        input_gate_logit = inputs @ self.W_ii + state.T @ self.W_hi + self.b_i
        i_t = torch.tanh(input_gate_logit) * input_gate_logit
        forget_gate_logit = inputs @ self.W_if + state.T @ self.W_hf + self.b_f
        f_t = torch.tanh(forget_gate_logit) * forget_gate_logit
        cell_gate_logit = inputs @ self.W_ig + state.T @ self.W_hg + self.b_g
        g_t = torch.tanh(cell_gate_logit)
        i_t = i_t.squeeze(-1)
        f_t = f_t.squeeze(-1)
        g_t = g_t.squeeze(-1)
        c_t = (1 - f_t) * c_prev + i_t * g_t
        return torch.stack([i_t, f_t, c_t], dim=0)


class MultiheadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size: int, num_heads: int,
                 batch_first: bool = True, bias: bool = False):
        super(MultiheadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        output = self.fc(output)
        return output, attention_weights
