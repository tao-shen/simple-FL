"""
FedL2O_Mamba：基于学习优化的联邦学习算法（Mamba 优化器，独立实现）。
与 fedl2o、fedl2o_lstm 为三种并列方法，本文件使用 Mamba 块替代 LSTM，支持长序列与状态传递。
"""
from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# 常量（与 LSTM 版本对齐）
DEFAULT_INPUT_SIZE = 2
DEFAULT_HIDDEN_SIZE = 30
DEFAULT_MAMBA_D_MODEL = 32
DEFAULT_MAMBA_D_STATE = 16
DEFAULT_MAMBA_D_CONV = 4
DEFAULT_MAMBA_EXPAND = 2
DEFAULT_CHUNK_SIZE = 512  # 每块处理的参数数量，避免长序列 OOM
DEFAULT_TRAINING_INTERVAL = 3
DEFAULT_META_LEARNING_RATE = 0.001
DEFAULT_AGGREGATOR_TRAINING_EPOCHS = 5
DEFAULT_GRADIENT_SCALE_PARAM = 10


# ==================== 纯 PyTorch 简易 Mamba 块（无 CUDA 依赖）====================

class MambaBlock(nn.Module):
    """
    简易 Mamba 选择性状态空间块，纯 PyTorch 实现，不依赖 mamba_ssm/CUDA。
    输入 (B, L, D)，输出 (B, L, D)；支持传入/返回状态以做分块推理。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(-torch.log(1 - torch.exp(-dt)))

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _causal_cut(self, x: torch.Tensor, L: int) -> torch.Tensor:
        return x[..., :L]

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x: (B, L, D)
        B, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Causal conv: (B, L, d_inner) -> (B, d_inner, L)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = self._causal_cut(x_conv, L)
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)  # (B, L, d_inner)

        # Selective: dt, B, C from x_conv
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        dt, BC = x_dbl.split([self.dt_rank, self.d_state * 2], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight, self.dt_proj.bias)
        dt = F.softplus(dt)
        B_sel, C_sel = BC.chunk(2, dim=-1)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # 简化的 selective scan：对时间步循环（纯 PyTorch，可后续换 CUDA kernel）
        y_list = []
        h = ssm_state  # (B, d_inner, d_state) or None
        for t in range(L):
            xt = x_conv[:, t, :]   # (B, d_inner)
            dt_t = dt[:, t, :]     # (B, d_inner)
            B_t = B_sel[:, t, :]   # (B, d_state)
            C_t = C_sel[:, t, :]   # (B, d_state)
            if h is None:
                h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            dA = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
            h = h * dA + xt.unsqueeze(-1) * dB
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * xt  # (B, d_inner)
            y_list.append(y_t)
        y = torch.stack(y_list, dim=1)  # (B, L, d_inner)
        h_last = h

        y = y * self.act(z)
        out = self.out_proj(y)  # (B, L, d_model)
        return out, conv_state, h_last


# ==================== L2A（与 LSTM 版本共用逻辑）====================

def _preprocess_gradient(gradient: torch.Tensor, scale_param: float = DEFAULT_GRADIENT_SCALE_PARAM) -> torch.Tensor:
    gradient_norm = torch.clamp(torch.log(torch.abs(gradient) + 1e-8) / scale_param, min=-1)
    gradient_sign = torch.clamp(
        torch.exp(torch.tensor(scale_param, device=gradient.device)) * gradient, min=-1, max=+1
    )
    return torch.stack([gradient_norm, gradient_sign], dim=-1)


# ==================== L2U_Mamba：用 Mamba 学习参数更新 ====================

class L2U_Mamba(nn.Module):
    """
    使用 Mamba 块学习模型参数更新；按块处理长参数序列，状态在块间传递。
    输出层状态仍为 [i, f, c]（学习率、权重衰减、模型参数），与 LSTM 版本语义一致。
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        args,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.input_proj = nn.Linear(2, d_model)
        self.mamba = MambaBlock(d_model, d_state=DEFAULT_MAMBA_D_STATE, d_conv=DEFAULT_MAMBA_D_CONV, expand=DEFAULT_MAMBA_EXPAND)
        self.output_proj = nn.Linear(d_model, 3)

    def forward(
        self,
        input_features: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # input_features: (param_size, 2)
        # hidden_states: [conv_state, ssm_state, output_state]
        #   output_state: (3, param_size) -> [i, f, c]
        param_size = input_features.shape[0]
        device = input_features.device

        if hidden_states is None or len(hidden_states) != 3:
            output_state = hidden_states[-1] if (hidden_states and len(hidden_states) >= 1) else None
            if output_state is None or output_state.shape != (3, param_size):
                output_state = torch.zeros(3, param_size, device=device, dtype=input_features.dtype)
            conv_state = None
            ssm_state = None
            hidden_states = [conv_state, ssm_state, output_state]
        else:
            conv_state, ssm_state, output_state = hidden_states

        n_chunks = (param_size + self.chunk_size - 1) // self.chunk_size
        updated_i = []
        updated_f = []
        updated_c = []

        for k in range(n_chunks):
            start = k * self.chunk_size
            end = min(start + self.chunk_size, param_size)
            chunk_len = end - start
            x_chunk = input_features[start:end]  # (chunk_len, 2)
            state_chunk = output_state[:, start:end]  # (3, chunk_len)
            i_prev, f_prev, c_prev = state_chunk[0], state_chunk[1], state_chunk[2]

            x_proj = self.input_proj(x_chunk).unsqueeze(0)  # (1, chunk_len, d_model)
            y_mamba, conv_state, ssm_state = self.mamba(x_proj, conv_state, ssm_state)
            y_mamba = y_mamba.squeeze(0)  # (chunk_len, d_model)
            ifc = self.output_proj(y_mamba)  # (chunk_len, 3)
            i_t = torch.tanh(ifc[:, 0]) * ifc[:, 0]
            f_t = torch.tanh(ifc[:, 1]) * ifc[:, 1]
            g_t = torch.tanh(ifc[:, 2])
            c_new = (1 - f_t) * c_prev + i_t * g_t

            updated_i.append(i_t)
            updated_f.append(f_t)
            updated_c.append(c_new)

        i_out = torch.cat(updated_i, dim=0)
        f_out = torch.cat(updated_f, dim=0)
        c_out = torch.cat(updated_c, dim=0)
        new_output_state = torch.stack([i_out, f_out, c_out], dim=0)
        updated_hidden_states = [conv_state, ssm_state, new_output_state]
        return c_out, updated_hidden_states


# ==================== L2O_Mamba ====================

class L2O_Mamba(nn.Module):
    """
    L2O (Mamba 版本)：L2A 聚合 + L2U_Mamba 更新。
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, args):
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.d_model = DEFAULT_MAMBA_D_MODEL
        self.l2u = L2U_Mamba(input_size, self.d_model, args, chunk_size=DEFAULT_CHUNK_SIZE)

    def forward(
        self,
        client_gradients: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        averaged_gradients = torch.mean(client_gradients, dim=-1) / self.args.lr_l
        processed_input = _preprocess_gradient(averaged_gradients)
        updated_params, updated_hidden_states = self.l2u.forward(processed_input, hidden_states)
        return updated_params, updated_hidden_states


# ==================== FedL2O_Mamba 主类 ====================

class FedL2O_Mamba(FedAvgM):
    """
    FedL2O (Mamba 版本)：使用 Mamba 作为 L2O 优化器网络。
    接口与 FedL2O_LSTM 一致，仅内部优化器由 LSTM 换为 Mamba。
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

        self.l2o_optimizer = L2O_Mamba(
            input_size, hidden_size, output_size, num_layers, self.args
        ).to(self.args.device)

        if not self.training_mode:
            model_path = "./assets/" + "_".join(
                ["aggr_mamba", self.args.dataset, self.args.iid, str(self.args.seed)]
            ) + ".pt"
            try:
                self.l2o_optimizer = torch.load(model_path, map_location="cpu").to(self.args.device)
            except FileNotFoundError:
                pass

        optimizer_factory = {"sgd": torch.optim.SGD, "adam": torch.optim.AdamW}
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
        model_parameters = self.server.model.to_vector().data.to(self.args.device)
        param_size = model_parameters.numel()
        weight_decay = torch.zeros_like(model_parameters)
        learning_rate = torch.zeros_like(model_parameters)
        output_state = torch.stack([weight_decay, learning_rate, model_parameters], dim=0)
        self.hidden_states = [None, None, output_state]
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
            self.hidden_states = [s.data if s is not None else None for s in updated_hidden_states[:2]]
            self.hidden_states.append(updated_hidden_states[2].data)
            self.server.model.from_vector(update_vector.data.cpu())

        self._record_learned_parameters()

    def _collect_client_gradients(self) -> torch.Tensor:
        client_gradients = [delta_model.to_vector() for delta_model in self.delta_models]
        return torch.stack(client_gradients).data.T.to(self.args.device)

    def _save_training_checkpoints(self, model_parameters: torch.Tensor, client_gradients: torch.Tensor):
        self.model_checkpoints.append(model_parameters.cpu())
        self.gradient_checkpoints.append(client_gradients.cpu())

    def _clear_training_checkpoints(self):
        self.model_checkpoints = []
        self.gradient_checkpoints = []

    def _record_learned_parameters(self):
        try:
            output_state = self.hidden_states[-1]
            learning_rate_state = output_state[1].data
            weight_decay_state = output_state[0].data
            self.recorder({
                'learned_lr_g': torch.mean(learning_rate_state ** 2).item(),
                'learned_w_decay': torch.mean(weight_decay_state ** 2).item(),
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
            shuffle=True,
        )
        for epoch in range(DEFAULT_AGGREGATOR_TRAINING_EPOCHS):
            for batch in train_loader:
                total_loss = 0.0
                batch = to_device(batch, self.args.device)
                hidden_states = copy.deepcopy(self.truncated_hidden_states)
                for client_gradients in self.gradient_checkpoints:
                    client_gradients = client_gradients.to(self.args.device)
                    updated_params, hidden_states = self.l2o_optimizer(client_gradients, hidden_states)
                    self._set_model_parameters_from_vector(surrogate_model, updated_params)
                    predictions = surrogate_model(batch)
                    labels = batch["label"]
                    total_loss += loss_function(predictions, labels)
                total_loss /= len(self.model_checkpoints)
                total_loss.backward()
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                print(f"Epoch {epoch}, Loss: {total_loss.data.item():.6f}")
        self.server.model.from_vector(updated_params.cpu())
        self.hidden_states = [s.data if s is not None else None for s in hidden_states[:2]]
        self.hidden_states.append(hidden_states[2].data)
        self.truncated_hidden_states = copy.deepcopy(self.hidden_states)

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
