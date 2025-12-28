from .fedavgm import FedAvgM
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class Elastic(FedAvgM):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)

    def clients_update(self):
        """
        客户端更新：执行局部训练并记录模型和参数敏感性的差异。
        """
        self.delta_models = []
        self.sensitivities = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model, sensitivity = self.local_update(self.clients[k], self.server.model)
            delta_model = self.diff_model(model, self.server.model)  # 模型差异
            self.delta_models.append(delta_model)
            self.sensitivities.append(sensitivity)  # 保存客户端的参数敏感性

    def local_update(self, client, model_g):
        """
        局部更新：在客户端上进行模型训练并计算参数敏感性。
        参数:
        - client: 当前客户端
        - model_g: 服务端模型
        返回:
        - model_l: 更新后的客户端模型
        - sensitivity: 参数敏感性
        """
        model_l = copy.deepcopy(model_g).to(self.args.device)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)

        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # 跳过没有数据的客户端
            return model_l, None

        sensitivity = {k: torch.zeros_like(v) for k, v in model_l.state_dict().items()}  # 初始化参数敏感性
        for batch in train_loader:
            optimizer.zero_grad()
            batch = to_device(batch, self.args.device)
            output = model_l(batch)
            label = batch['label']
            loss = model_l.loss_fn(output, label)
            loss.backward()

            # 累积梯度的绝对值作为敏感性
            for name, param in model_l.named_parameters():
                sensitivity[name] += torch.abs(param.grad)

            optimizer.step()

        # 使用指数移动平均平滑敏感性
        for key in sensitivity:
            sensitivity[key] = self.args.mu * sensitivity[key] + (1 - self.args.mu) * sensitivity[key]
        
        return model_l, sensitivity

    def server_update(self):
        """
        服务端更新：基于客户端模型和敏感性执行弹性聚合。
        """
        aggregated_model = copy.deepcopy(self.server.model)
        aggregated_params = aggregated_model.state_dict()

        # 获取每个参数的最大敏感性
        # Filter out None sensitivities and set them to zeros
        valid_sensitivities = []
        for s in self.sensitivities:
            if s is not None:
                valid_sensitivities.append(s)
            else:
                zeros = {k: torch.zeros_like(v).to(self.args.device) for k,v in self.server.model.state_dict().items()}
                valid_sensitivities.append(zeros)
        self.sensitivities = valid_sensitivities
        max_sensitivity = {k: torch.max(torch.stack([s[k] for s in self.sensitivities])) for k in self.sensitivities[0]}
        
        with torch.no_grad():
            for name, param in aggregated_params.items():
                # 弹性聚合：根据敏感性调整参数更新
                adaptive_update = torch.zeros_like(param)
                for model, sensitivity in zip(self.delta_models, self.sensitivities):
                    # weight = self.args.client_fraction / len(self.delta_models)
                    adaptive_coefficient = 1 + 0.1 - sensitivity[name] / max_sensitivity[name]
                    adaptive_update += 0.1 * adaptive_coefficient.cpu() * model.state_dict()[name].cpu()
                aggregated_params[name] -= self.args.lr_g * adaptive_update

        self.server.model.load_state_dict(aggregated_params)
