from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedAvg(FL):

    def __init__(self, server, clients, args):
        self.server = server
        self.clients = clients
        self.args = args
        self.N = len(self.clients)
        self.T = args.communication_rounds
        self.C = args.client_fraction
        self.E = args.local_epochs
        if args.clients_per_round is None:
            self.K = int(self.C*self.N)
        else:
            self.K = args.clients_per_round
        if args.FL_validate_clients == True:
            n=np.arange(self.N)
            self.clients_test_ind = np.random.choice(
            n, int(args.validate_client_ratio*self.N), replace=False)
            self.clients_train_ind = np.setdiff1d(n, self.clients_test_ind)
        else:
            self.clients_train_ind = np.arange(self.N)
        
        self.opts = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}

    def server_init(self):
        self.server.model = self.server.init_model()

    def candidates_sampling(self):
        if self.args.FL_validate_clients == True:
            candidate_clients = self.clients_test_ind
        else:
            candidate_clients = self.clients_train_ind
        self.candidates = np.random.choice(
            candidate_clients,
            min(self.K, len(candidate_clients)),
            replace=False,
        )
        # 这里是为了保证挑出一部分test_client后，剩下的client可能小于K。

    def clients_update(self):
        self.models = []
        for k in tqdm(self.candidates, desc=self.__class__.__name__):
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())

    def server_update(self):
        w = self.averaging(self.models)
        self.server.model.load_state_dict(w)

    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        for E in range(self.E):
            model_l.fit(train_loader, optimizer)
        return model_l

    def averaging(self, models, w=None):
        if w is None:
            w = self.args.avg
        with torch.no_grad():
            if w == 'w':
                num_sample = torch.tensor([m.train_num for m in models])
                num_sum = torch.sum(num_sample)
                if num_sum ==0:
                    weights = 1/len(models)
                else:
                    weights = num_sample[:, None]/num_sum
            else:
                weights = 1/len(models)
            size = {k: v.size() for k, v in models[0].named_parameters()}
            feats = self.to_feats(models)
            vector = self.feats2vector(feats)
            w_avg = torch.sum(weights*vector, dim=0)
            w_dict = self.vector2model_dict(w_avg, size)
        return w_dict

    def evaluate(self):
        self.server.model.evaluate(
            self.server.test_loader, user_offset=self.server.init.user_offset['test'], recorder=self.args.recorder, ifprint=True)

    def to_feats(self, models):
        feats = {}
        for k in models[0].state_dict().keys():
            dicts = [model.state_dict()[k].unsqueeze(0) for model in models]
            v = torch.cat(tuple(dicts), dim=0)
            k = k.replace('.', '-')
            feats[k] = v
        return feats

    def feats2vector(self, feats):
        # Flattening the parameters
        # param_shapes = {k: v.shape[1:] for k, v in feats.items()}
        flattened_params = torch.hstack(
            [p.flatten(1) for p in feats.values()])
        return flattened_params

    def vector2model_dict(self, flattened_params, param_shapes):

        # Reshaping the flattened parameters
        start_idx = 0
        model_dict = {}
        for key, shape in param_shapes.items():
            size = np.prod(shape)
            model_dict[key] = flattened_params[start_idx:start_idx +
                                               size].reshape(shape)
            start_idx += size
        return model_dict

    def recorder(self, kwargs):
        for k,v in kwargs.items():
            if k not in self.args.recorder:
                self.args.recorder[k]=[]
            self.args.recorder[k].append(v)

    def training(self):
        for t in range(self.T):
            print('==========the {}-th round==========='.format(t))
            self.candidates_sampling()
            self.clients_update()
            self.server_update()
            self.evaluate()

    def zero_weights(self, model):
        for n, p in model.named_parameters():
            p.data.zero_()        

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
