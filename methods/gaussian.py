from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import pickle


class Gaussian(FL):

    def __init__(self, server, clients, args):
        self.server = server
        self.clients = clients
        self.args = args
        self.N = len(self.clients)
        self.T = args.communication_rounds
        self.C = args.client_fraction
        self.E = args.local_epochs
        if args.clients_per_round is None:
            self.K = int(self.C * self.N)
        else:
            self.K = args.clients_per_round

    def server_init(self):
        self.server.model = self.server.init_model()

    def candidates_sampling(self):
        self.candidates = np.random.choice(np.arange(self.N), self.K, replace=False)

    def clients_update(self):
        self.models = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())

    def server_update(self):
        m = dict(copy.deepcopy(self.server.model.state_dict()))
        grad_g = self.one_step_on_global_data(self.server.model)
        grad_s = self.one_step_on_sampled_data(self.server.model)
        grad_t = self.one_step_on_client_data(self.server.model)
        # c = self.models[0].state_dict()
        w = self.averaging(self.models)
        # grad_t = {
        #     k: (m[k] - c[k]) / self.args.lr_l / self.args.clients_per_round
        #     for k in w.keys()
        # }

        self.server.model.load_state_dict(w)
        # 立即保存模型状态
        with open(
            "{}_{}_{}_client_one.pkl".format(
                self.args.dataset, "lenet5", self.args.iid
            ),
            "ab",
        ) as file:
            pickle.dump({"grad_g": grad_g, "grad_s": grad_s, "grad_t": grad_t}, file)

    def one_step_on_client_data(self, model):
        device = self.args.device
        model_l = copy.deepcopy(model).to(device)

        # 选择优化器
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        optimizer = opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )

        for i in self.candidates:
            if len(self.clients[i].train_data) > 0:
                choose = i
                break
        train_data = np.concatenate([self.clients[choose].train_data])
        np.random.shuffle(train_data)
        try:
            # 尝试创建数据加载器，如果客户端没有数据则跳过
            train_loader = DataLoader(
                Dataset(train_data, self.args),
                batch_size=len(train_data),
                shuffle=True,
            )
        except:  # 跳过没有数据的客户端
            return model_l

        total_steps = 1
        current_step = 0
        data_iterator = iter(train_loader)

        while current_step < total_steps:
            try:
                # 获取下一批数据
                batch = next(data_iterator)
            except StopIteration:
                # 如果数据耗尽，则重启迭代器
                data_iterator = iter(train_loader)
                batch = next(data_iterator)

            # 确保数据在正确的设备上
            batch = to_device(batch, device)

            # 执行单步训练
            optimizer.zero_grad()
            output = model_l(batch)
            label = batch["label"]
            loss = model_l.loss_fn(output, label)
            loss.backward()
            # optimizer.step()

            current_step += 1
        grad = {name: param.grad for name, param in model_l.named_parameters()}

        return grad

    def one_step_on_sampled_data(self, model):
        device = self.args.device
        model_l = copy.deepcopy(model).to(device)

        # 选择优化器
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        optimizer = opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )

        train_data = np.concatenate(
            [self.clients[i].train_data for i in self.candidates]
        )
        np.random.shuffle(train_data)
        try:
            # 尝试创建数据加载器，如果客户端没有数据则跳过
            train_loader = DataLoader(
                Dataset(train_data, self.args),
                batch_size=len(train_data),
                shuffle=True,
            )
        except:  # 跳过没有数据的客户端
            return model_l

        total_steps = 1
        current_step = 0
        data_iterator = iter(train_loader)

        while current_step < total_steps:
            try:
                # 获取下一批数据
                batch = next(data_iterator)
            except StopIteration:
                # 如果数据耗尽，则重启迭代器
                data_iterator = iter(train_loader)
                batch = next(data_iterator)

            # 确保数据在正确的设备上
            batch = to_device(batch, device)

            # 执行单步训练
            optimizer.zero_grad()
            output = model_l(batch)
            label = batch["label"]
            loss = model_l.loss_fn(output, label)
            loss.backward()
            # optimizer.step()

            current_step += 1
        grad = {name: param.grad for name, param in model_l.named_parameters()}

        return grad

    def one_step_on_global_data(self, model):
        batchsize = 1000
        device = self.args.device
        model_l = copy.deepcopy(model).to(device)

        # 选择优化器
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        optimizer = opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )

        train_data, _ = self.server.init.data
        np.random.shuffle(train_data)
        # 尝试创建数据加载器，这次使用较小的batch_size
        train_loader = DataLoader(
            Dataset(train_data, self.args),
            batch_size=batchsize,  # 使用预定义的合理大小的batch_size
            shuffle=True,
        )

        optimizer.zero_grad()  # 初始化梯度
        for batch in tqdm(train_loader):
            # 确保数据在正确的设备上
            batch = to_device(batch, device)

            # 计算模型输出和损失
            output = model_l(batch)
            label = batch["label"]
            loss = model_l.loss_fn(output, label)

            # 计算梯度，但是不立即执行优化步骤
            loss.backward()

        # 在处理所有批次后执行单个优化步骤
        # optimizer.step()
        grad = {
            name: param.grad * batchsize / len(train_data)
            for name, param in model_l.named_parameters()
        }

        return grad

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        optimizer = opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args),
                batch_size=self.args.local_batch_size,
                shuffle=True,
            )
        except:  # skip dataless client
            return model_l
        for E in range(100):
            self.fixed_steps(model_l, train_loader, optimizer)
            model_l.fit(train_loader, optimizer)
        return model_l

    def local_update(self, client, model_g):
        # 深拷贝模型，并迁移到正确的设备
        device = self.args.device
        model_l = copy.deepcopy(model_g).to(device)

        # 选择优化器
        opts = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
        optimizer = opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay
        )

        try:
            # 尝试创建数据加载器，如果客户端没有数据则跳过
            train_loader = DataLoader(
                Dataset(client.train_data, self.args),
                batch_size=self.args.local_batch_size,
                shuffle=True,
            )
        except:  # 跳过没有数据的客户端
            return model_l

        total_steps = 10
        current_step = 0
        data_iterator = iter(train_loader)

        while current_step < total_steps:
            try:
                # 获取下一批数据
                batch = next(data_iterator)
            except StopIteration:
                # 如果数据耗尽，则重启迭代器
                data_iterator = iter(train_loader)
                batch = next(data_iterator)

            # 确保数据在正确的设备上
            batch = to_device(batch, device)

            # 执行单步训练
            optimizer.zero_grad()
            output = model_l(batch)
            label = batch["label"]
            loss = model_l.loss_fn(output, label)
            loss.backward()
            optimizer.step()

            current_step += 1

        return model_l

    def averaging(self, models):
        with torch.no_grad():
            if self.args.avg == "w":
                num_sample = torch.tensor([m.train_num for m in models])
                num_sum = torch.sum(num_sample)
                if num_sum == 0:
                    weights = 1 / len(models)
                else:
                    weights = num_sample[:, None] / num_sum
            else:
                weights = 1 / len(models)
            size = {k: v.size() for k, v in models[0].named_parameters()}
            feats = self.to_feats(models)
            vector = self.feats2vector(feats)
            w_avg = torch.sum(weights * vector, dim=0)
            w_dict = self.vector2model_dict(w_avg, size)
        return w_dict

    def evaluate(self):
        self.server.model.evaluate(
            self.server.test_loader,
            user_offset=self.server.init.user_offset["test"],
            recorder=self.args.recorder,
            ifprint=True,
        )

    def to_feats(self, models):
        feats = {}
        for k in models[0].state_dict().keys():
            dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
            v = torch.cat(tuple(dicts), dim=0)
            k = k.replace(".", "-")
            feats[k] = v
        return feats

    def feats2vector(self, feats):
        # Flattening the parameters
        # param_shapes = {k: v.shape[1:] for k, v in feats.items()}
        flattened_params = torch.hstack([p.flatten(1) for p in feats.values()])
        return flattened_params

    def vector2model_dict(self, flattened_params, param_shapes):

        # Reshaping the flattened parameters
        start_idx = 0
        model_dict = {}
        for key, shape in param_shapes.items():
            size = np.prod(shape)
            model_dict[key] = flattened_params[start_idx : start_idx + size].reshape(
                shape
            )
            start_idx += size
        return model_dict

    # def averaging1(self, models):
    #     with torch.no_grad():
    #         model = copy.deepcopy(models[0])
    #         w = model.state_dict()
    #         num_sample = torch.tensor([m.train_num for m in models])
    #         num_sum = torch.sum(num_sample)
    #         for key in w.keys():
    #             w[key] = 0
    #             if self.args.avg == 'w':
    #                 for m in models:
    #                     w[key] += m.state_dict()[key]*m.train_num
    #                 w[key] /= num_sum
    #             else:
    #                 for m in models:
    #                     w[key] += m.state_dict()[key]
    #                 w[key] /= len(models)
    #     return w
