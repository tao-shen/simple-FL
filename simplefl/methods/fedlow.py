import torch
import torch.nn as nn
import copy
from .fedavg import FedAvg
from .fl import *
from simplefl.utils import *
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. FedLow will not work without it.")


class FedLow(FedAvg):
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
        # 设置LoRA参数的默认值
        self.rank = getattr(args, 'lora_rank', 16)  # LoRA秩，默认为16
        # self.rank = 32
        self.lora_alpha = getattr(args, 'lora_alpha', 32)  # LoRA alpha参数，默认为32
        self.lora_dropout = getattr(args, 'lora_dropout', 0.1)  # LoRA dropout率，默认为0.1

    def add_lora_to_model(self, model):
        # 获取所有带训练参数的模块名称
        target_modules = [name for name, module in model.named_modules() if list(module.parameters(recurse=False))]
        
        # 配置LoRA
        config = LoraConfig(
            r=self.rank,
            # lora_alpha=self.lora_alpha,
            # lora_dropout=self.lora_dropout,
            # task_type=TaskType.FEATURE_EXTRACTION,
            # inference_mode=False,
            target_modules=target_modules,  # 使用获取到的所有带训练参数的模块
            bias="lora_only",
            use_rslora=True
        )
        # 返回添加了LoRA层的模型
        return get_peft_model(model, config)

    def local_update(self, client, model_g):
        # 创建全局模型的本地副本
        model_l = copy.deepcopy(model_g)
        # 为本地模型添加LoRA层
        model_l = self.add_lora_to_model(model_l)
        
        # 仅优化LoRA参数
        optimizer = self.opts[self.args.local_optimizer](model_l.parameters(), lr=self.args.lr_l)

        try:
            # 创建训练数据加载器
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # 跳过没有数据的客户端
            return model_l

        # 本地训练循环
        for E in range(self.E):
            model_l.fit(train_loader, optimizer)
        return model_l

    def clients_update(self):
        # 收集所有客户端的LoRA参数
        self.loras = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            lora_updates = {name: param.data.cpu() for name, param in model.named_parameters() if param.requires_grad}
            self.loras.append(lora_updates)

    def server_update(self):
        # 聚合参数并更新全局模型
        avg_params = self.average_params(self.loras)
        self.apply_params_to_global_model(self.server.model, avg_params)

    def average_params(self, params_list):
        global_update = {}
        for key in params_list[0].keys():
            if 'bias' in key:
                # 对于偏置层，直接平均
                global_update[key] = sum(params[key] for params in params_list) / len(params_list)
            elif 'weight' in key:
                if 'lora_A' in key:
                    lora_B_key = key.replace('lora_A', 'lora_B')
                    original_key = key.replace('base_model.model.', '').replace('.lora_A.default.weight', '.weight')
                    if 'conv' in key.lower():
                        # 对于卷积层，使用 einsum 合并
                        update = torch.zeros_like(self.server.model.state_dict()[original_key])
                        for params in params_list:
                            update += torch.einsum('ocab,cixy->oixy', params[lora_B_key], params[key])
                    else:
                        # 对于线性层，使用矩阵乘法
                        update = torch.zeros_like(self.server.model.state_dict()[original_key])
                        for params in params_list:
                            update += torch.matmul(params[lora_B_key], params[key])
                    global_update[original_key] = update / len(params_list)
                elif 'lora_B' not in key:
                    # 对于非 LoRA 权重，直接平均
                    global_update[key] = sum(params[key] for params in params_list) / len(params_list)
        return global_update

    def apply_params_to_global_model(self, model, global_update):
        with torch.no_grad():
            for key, update in global_update.items():
                if key in model.state_dict():
                    model.state_dict()[key].add_(update)

# LoRALayer类已被移除，因为不再需要
