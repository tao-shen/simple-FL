"""
FedDyn (Federated Dynamic Regularization) 方法实现

FedDyn 通过动态正则化项来解决联邦学习中的客户端漂移问题。
核心思想是在客户端本地训练时，添加一个动态正则化项，使客户端模型向全局模型靠拢，
同时通过梯度近似项来补偿正则化带来的偏差。

重要参数说明：
1. alpha: 正则化强度参数（核心参数）
   - 含义：控制动态正则化项的强度，影响客户端模型与全局模型的接近程度
   - alpha 越大：正则化越强，客户端模型被强制拉向全局模型，收敛快但可能过度约束
   - alpha 越小：正则化越弱，客户端有更多探索空间，但可能导致客户端漂移
   - 默认值：0.5
   - 调参建议：
     * 如果前期收敛快但后期性能下降：减小 alpha（如 0.1, 0.05, 0.01）
     * 如果收敛慢或客户端漂移严重：增大 alpha（如 1.0, 2.0）
     * 可以考虑自适应策略：随训练轮数逐渐减小 alpha

2. lr_l: 本地学习率
   - 含义：客户端本地训练时的学习率
   - lr_l 越大：本地训练步长越大，收敛快但可能不稳定
   - lr_l 越小：本地训练步长越小，更稳定但收敛慢
   - 默认值：0.001
   - 调参建议：通常与 alpha 配合调整，alpha 较小时可适当增大 lr_l

3. weight_decay: 权重衰减（L2正则化）
   - 含义：防止模型过拟合的正则化项
   - weight_decay 越大：模型参数越小，防止过拟合但可能欠拟合
   - weight_decay 越小：模型参数更大，可能过拟合
   - 默认值：由 args.weight_decay 决定

4. local_epochs (E): 本地训练轮数
   - 含义：每个客户端每轮参与训练时，在本地数据上训练的 epoch 数
   - E 越大：本地训练更充分，但客户端漂移风险增加
   - E 越小：本地训练不充分，但客户端更接近全局模型
   - 默认值：由 args.local_epochs 决定
"""

from simplefl.methods.fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch

class FedDyn(FedAvg):
    """
    FedDyn (Federated Dynamic Regularization) 方法实现
    
    核心思想：通过动态正则化项和梯度近似来减少客户端漂移，提高收敛性能。
    
    重要参数说明：
    1. alpha: 正则化强度参数（核心超参数）
       - 含义：控制客户端模型与全局模型之间的约束强度
       - alpha 越大：正则化越强，客户端模型被强制靠近全局模型，收敛快但可能过度约束
       - alpha 越小：正则化越弱，客户端有更多探索空间，但可能导致客户端漂移
       - 默认值：0.5
       - 调参建议：
         * 如果前期收敛快但后期性能下降：减小 alpha（如 0.1, 0.05, 0.01）
         * 如果收敛慢或不稳定：增大 alpha（如 0.5, 1.0, 2.0）
    
    2. lr_l: 本地学习率
       - 含义：客户端本地训练时的学习率
       - lr_l 越大：本地更新步长越大，训练更快但可能不稳定
       - lr_l 越小：本地更新步长越小，训练更稳定但可能较慢
       - 默认值：0.001
       - 调参建议：通常与 alpha 配合调整，alpha 较小时可适当增大 lr_l
    
    3. weight_decay: 权重衰减（L2正则化）
       - 含义：防止模型过拟合的正则化项
       - weight_decay 越大：模型参数被约束得越小，防止过拟合但可能欠拟合
       - weight_decay 越小：模型参数约束小，可能过拟合
       - 调参建议：根据数据集复杂度调整，通常 0.0001-0.01
    
    4. local_epochs (E): 本地训练轮数
       - 含义：每个客户端每轮参与训练时，在本地数据上训练的 epoch 数
       - local_epochs 越大：本地训练更充分，但客户端漂移风险增加
       - local_epochs 越小：客户端漂移风险小，但本地训练可能不充分
       - 调参建议：通常 1-10，需要与 alpha 平衡
    """
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
    
    def server_init(self, alpha=0.5):
        """
        初始化服务器端模型和 FedDyn 相关参数
        
        Args:
            alpha (float): 正则化强度参数（核心超参数），默认值为 0.5
                - alpha 越大：正则化越强，客户端模型被强制靠近全局模型
                  * 优点：收敛快，客户端漂移小
                  * 缺点：可能过度约束，后期性能下降，难以找到最优解
                - alpha 越小：正则化越弱，客户端有更多探索空间
                  * 优点：后期有更多探索空间，可能找到更好的解
                  * 缺点：收敛可能较慢，客户端漂移风险增加
                - 调参建议：
                  * 如果前期收敛快但后期性能下降：减小 alpha（如 0.1, 0.05, 0.01）
                  * 如果收敛慢或客户端漂移严重：增大 alpha（如 1.0, 2.0）
                  * 可以通过配置文件 methods.yaml 设置，或修改此默认值
        """
        self.server.model = self.server.init_model()  
        self.alpha = alpha  # 正则化强度参数（核心超参数）
        
        # h: 初始化动态正则化项（用于服务器更新）
        # h 是一个与模型结构相同的零参数张量，用于存储累积的梯度近似
        # 在 server_update 中会更新：h = h - alpha * (u_avg - m)
        h = copy.deepcopy(self.server.model)
        self.zero_weights(h)
        self.h = h.state_dict()
        
    def server_update(self):
        """
        服务器更新方法
        
        FedDyn 的服务器更新包含两个关键步骤：
        1. 更新动态正则化项 h：h = h - alpha * (u_avg - m)
           其中 u_avg 是客户端模型参数的加权平均，m 是上一轮全局模型参数
        2. 更新全局模型：w = w_avg - (1/alpha) * h
           其中 w_avg 是客户端模型参数的加权平均
        
        注意：alpha 越大，h 的更新幅度越大，全局模型受 h 的影响也越大（因为 1/alpha 越小）
        """
        # u_avg: 客户端模型参数的加权平均（用于更新 h）
        u_avg = self.averaging(self.models, w='u')
        # w_avg: 客户端模型参数的加权平均（用于更新全局模型）
        w_avg = self.averaging(self.models, w='w')
        # m: 上一轮的全局模型参数
        m = copy.deepcopy(self.server.model.state_dict())
        model_param = {}
        
        for k, v in self.h.items():
            # 更新动态正则化项 h
            # h[k] = h[k] - alpha * (u_avg[k] - m[k])
            # alpha 越大：h 的更新幅度越大，对客户端漂移的惩罚越强
            self.h[k] = v - self.alpha * (u_avg[k] - m[k])
            
            # 更新全局模型参数
            # model_param[k] = w_avg[k] - (1/alpha) * h[k]
            # alpha 越大：1/alpha 越小，h 对全局模型的影响越小
            # alpha 越小：1/alpha 越大，h 对全局模型的影响越大
            model_param[k] = w_avg[k] - 1 / self.alpha * self.h[k]
        
        self.server.model.load_state_dict(model_param)
            

    def local_update(self, client, model_g):
        """
        客户端本地更新方法
        
        Args:
            client: 客户端对象
            model_g: 全局模型（从服务器接收）
        
        Returns:
            model_l: 更新后的本地模型
        
        重要参数说明：
        - lr_l (self.args.lr_l): 本地学习率
          * 越大：本地训练步长越大，收敛快但可能不稳定
          * 越小：本地训练步长越小，更稳定但收敛慢
          * 默认值：0.001
          * 调参建议：通常与 alpha 配合调整，alpha 较小时可适当增大 lr_l
        
        - weight_decay (self.args.weight_decay): 权重衰减（L2正则化）
          * 越大：L2正则化越强，防止过拟合但可能欠拟合
          * 越小：L2正则化越弱，可能过拟合
          * 调参建议：根据数据集复杂度调整，通常 0.0001-0.01
        
        - local_epochs (self.E): 本地训练轮数
          * 越大：本地训练更充分，但客户端漂移风险增加
          * 越小：本地训练不充分，但客户端更接近全局模型
          * 默认值：由 args.local_epochs 决定
          * 调参建议：通常 1-10，需要与 alpha 平衡
        """
        model_l = copy.deepcopy(model_g)
        # prox: 存储上一轮全局模型的副本，用于计算动态正则化项
        # 这个参数在本地训练过程中保持不变，用于衡量当前模型与全局模型的差异
        prox = copy.deepcopy(model_g)
        
        # 初始化优化器
        # lr_l: 本地学习率，控制本地训练的步长
        #   - 越大：步长越大，训练更快但可能不稳定
        #   - 越小：步长越小，训练更稳定但可能较慢
        # weight_decay: 权重衰减，L2正则化系数
        #   - 越大：L2正则化越强，防止过拟合但可能欠拟合
        #   - 越小：L2正则化越弱，可能过拟合
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), 
                batch_size=self.args.local_batch_size, shuffle=True)
        except:  # 跳过没有数据的客户端
            return model_l
        
        # 进行 E 轮本地训练
        # E (local_epochs) 的影响：
        # - 越大：本地训练更充分，但可能导致客户端漂移
        # - 越小：本地训练不充分，但客户端更接近全局模型
        # 需要与 alpha 参数平衡：alpha 较大时可以适当增大 E，alpha 较小时应减小 E
        for E in range(self.E):
            self.fit_feddyn(model_l, train_loader, optimizer, prox, client)
        
        return model_l

    def fit_feddyn(self, model, train_loader, optimizer, prox, client):
        """
        执行 FedDyn 的本地训练步骤
        
        FedDyn 的损失函数包含三个部分：
        1. loss: 标准分类/回归损失
        2. prox_loss: 动态正则化项 = alpha * 0.5 * ||p - prox_param||^2
        3. grad_loss: 梯度近似项 = p * grad（用于补偿正则化带来的偏差）
        
        total_loss = loss + prox_loss + grad_loss
        
        参数影响：
        - alpha 越大：prox_loss 越大，模型被强制拉向全局模型（prox）
          * 优点：收敛快，客户端漂移小
          * 缺点：可能过度约束，后期性能下降
        - alpha 越小：prox_loss 越小，模型有更多探索空间
          * 优点：后期有更多探索空间，可能找到更好的解
          * 缺点：收敛可能较慢，客户端漂移风险增加
        """
        model.train().to(model.device)
        # 只获取状态字典，不将整个模型移到设备，避免显存积累
        prox_params = prox.state_dict()
        
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            output = model(batch)
            label = batch['label']
            
            # 1. 标准损失：分类/回归损失
            loss = model.loss_fn(output, label)
            
            # 2. 动态正则化项（核心部分）
            # prox_loss = alpha * 0.5 * ||p - prox_param||^2
            # 其中 p 是当前模型参数，prox_param 是上一轮全局模型参数
            # 
            # alpha 的影响：
            # - alpha 越大：正则化项越大，模型参数 p 被强制拉向 prox_param
            #   这会使客户端模型更接近全局模型，收敛快但可能过度约束
            # - alpha 越小：正则化项越小，模型参数 p 有更多探索空间
            #   这允许客户端找到更好的局部最优解，但可能导致客户端漂移
            prox_loss = 0.0
            for n, p in model.named_parameters():
                # 只在需要时将参数移到设备，使用完立即释放
                prox_param = prox_params[n].to(p.device)
                prox_loss += self.alpha * 0.5 * torch.norm(p - prox_param) ** 2
                del prox_param  # 显式释放
            
            # 3. 梯度近似项（用于补偿正则化带来的偏差）
            # grad_loss = p * grad
            # 其中 grad 是累积的梯度近似，用于补偿动态正则化项带来的偏差
            # 这个项确保 FedDyn 能够收敛到正确的全局最优解
            grad_loss = 0.0            
            if hasattr(client, 'grad'):
                for n, p in model.named_parameters():
                    # 只在需要时将梯度移到设备
                    grad = client.grad[n].to(p.device)
                    grad_loss += torch.sum(p * grad)
                    del grad  # 显式释放
                            
            # 总损失 = 标准损失 + 动态正则化项 + 梯度近似项
            total_loss = loss + prox_loss + grad_loss
            total_loss.backward()
            optimizer.step()

        # 更新本地梯度近似（用于下一轮训练）
        self.update_local_gradient_approximation(client, model, prox)

    def update_local_gradient_approximation(self, client, model, prox):
        """
        更新本地梯度近似
        
        这个方法维护每个客户端的梯度近似 grad，用于补偿动态正则化项带来的偏差。
        
        更新公式：
        - 如果已有 grad：grad = grad - alpha * (m - prox_param)
        - 如果首次初始化：grad = -alpha * (m - prox_param)
        
        其中：
        - m: 当前本地模型参数（训练后）
        - prox_param: 上一轮全局模型参数（训练前）
        - alpha: 正则化强度参数
        
        参数影响：
        - alpha 越大：grad 的更新幅度越大，对模型变化的响应越敏感
        - alpha 越小：grad 的更新幅度越小，对模型变化的响应越不敏感
        
        这个梯度近似项在 fit_feddyn 中用于计算 grad_loss，确保 FedDyn 能够
        收敛到正确的全局最优解，而不仅仅是局部最优解。
        """
        with torch.no_grad():
            prox_params = prox.state_dict()  # 获取CPU上的参数
            
            if hasattr(client, 'grad'):
                # 更新已有的梯度近似
                # grad = grad - alpha * (m - prox_param)
                # alpha 越大：grad 的更新幅度越大
                for k, m in model.named_parameters():
                    # 将参数移到CPU进行计算，避免显存积累
                    m_cpu = m.detach().cpu()
                    p_cpu = prox_params[k].cpu()
                    # 确保 client.grad[k] 也在 CPU 上
                    if client.grad[k].device.type != 'cpu':
                        client.grad[k] = client.grad[k].cpu()
                    client.grad[k] = client.grad[k] - self.alpha * (m_cpu - p_cpu)
                    del m_cpu, p_cpu  # 显式释放
            else:
                # 首次初始化梯度近似
                # grad = -alpha * (m - prox_param)
                # alpha 越大：初始 grad 的绝对值越大
                client.grad = {}
                for k, m in model.named_parameters():
                    # 将参数移到CPU进行存储，避免显存积累
                    m_cpu = m.detach().cpu()
                    p_cpu = prox_params[k].cpu()
                    client.grad[k] = -self.alpha * (m_cpu - p_cpu)
                    del m_cpu, p_cpu  # 显式释放
