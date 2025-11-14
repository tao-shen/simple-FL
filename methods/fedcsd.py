from .fedavg import FedAvg
from .fl import *
from utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F


class FedCSD(FedAvg):

    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
        self.alpha = 0.9  # momentum for teacher model update
        self.tau = 10  # temperature for distillation
        self.mu = 0.001  # weight for distillation loss

    def server_init(self):
        super().server_init()
        self.teacher_model = copy.deepcopy(self.server.model).to(self.args.device)
        self.global_prototype = None

    def candidates_sampling(self):
        super().candidates_sampling()
        self.global_prototype = self.compute_global_prototype()

    def compute_global_prototype(self):
        prototypes = []
        for k in self.candidates:
            prototype = self.compute_local_prototype(self.clients[k])
            prototypes.append(prototype)
        return torch.mean(torch.stack(prototypes), dim=0)

    def compute_local_prototype(self, client):
        self.teacher_model.eval()
        prototypes = []
        with torch.no_grad():
            for c in range(self.teacher_model.num_classes):
                class_data = client.train_data[client.train_data["label"] == c]
                if len(class_data["label"]) != 0:
                    class_data = {
                        k: torch.from_numpy(class_data[k])
                        for k in class_data.dtype.names
                    }
                    to_device(class_data, self.args.device)
                    logits = self.teacher_model(class_data)
                    prototype = logits.mean(dim=0)
                    prototypes.append(prototype)
                else:
                    prototypes.append(
                        torch.zeros(
                            self.teacher_model.num_classes, device=self.args.device
                        )
                    )
        return torch.stack(prototypes)

    def clients_update(self):
        self.models = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
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

        for E in range(self.E):
            self.fit_fedcsd(model_l, train_loader, optimizer)

        return model_l

    def fit_fedcsd(self, model, train_loader, optimizer):
        model.train().to(self.args.device)
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc="local_update")

        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, self.args.device)
            output = model(batch)
            label = batch["label"]

            ce_loss = model.loss_fn(output, label)
            csd_loss = self.compute_csd_loss(batch, label, model)

            loss = ce_loss + self.mu * csd_loss
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            loss_avg = loss_total / (idx + 1)
        model.train_num = len(train_loader.dataset)

    def compute_csd_loss(self, batch, label, model):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(batch)

        student_logits = model(batch)

        similarity = F.cosine_similarity(
            student_logits.unsqueeze(1),
            self.global_prototype.to(self.args.device),
            dim=2,
        )
        similarity = F.softmax(similarity, dim=1)

        refined_teacher_logits = similarity * teacher_logits.unsqueeze(1)
        refined_teacher_logits = refined_teacher_logits.sum(dim=1)

        mask = self.compute_adaptive_mask(teacher_logits, label)

        loss = -torch.mean(
            mask
            * self.tau**2
            * (
                F.softmax(refined_teacher_logits / self.tau, dim=1)
                * F.log_softmax(student_logits / self.tau, dim=1)
            ).sum(dim=1)
        )

        return loss

    def compute_adaptive_mask(self, logits, label):
        probs = F.softmax(logits, dim=1)
        correct_probs = probs[torch.arange(probs.size(0)), label]
        mask = (correct_probs > (1 / self.teacher_model.num_classes)).float()
        return mask

    def server_update(self):
        with torch.no_grad():
            super().server_update()
            self.update_teacher_model()

    def update_teacher_model(self):
        for t_param, g_param in zip(
            self.teacher_model.parameters(), self.server.model.parameters()
        ):
            t_param.data = self.alpha * t_param.data + (
                1 - self.alpha
            ) * g_param.data.to(t_param.device)

    def zero_weights(self, model):
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
