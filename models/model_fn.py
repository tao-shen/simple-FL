import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd, numpy as np
from scipy.special import kl_div
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class Container():
    def __init__(self, x):
        self.data = x
        self.len = len(list(self.data.values())[0])
        
    def __getitem__(self, idx):
        x = {}
        for key, value in self.data.items():
            x[key] = value[idx]
        return x

    def __len__(self):
        return self.len

def to_device(x, device):
    if isinstance(x, dict):
        for key, value in x.items():
            x[key] = value.to(device)
    elif isinstance(x, Container):
        x = to_device(x.data, device)
    # elif isinstance(x, Container_mcc):
    #     x = to_device(x.data, device)
    else:
        x = x.to(device)
    return x


# def to_cuda(x, device):
#     if isinstance(x, dict):
#         for key, value in x.items():
#             x[key] = value.cuda()
#     elif isinstance(x, Container):
#         x = to_device(x.data, device)
#     else:
#         x = x.cuda()
#     return x

class Model_fn():

    def __init__(self, args, **kwargs):
        # super().__init__()
        self.device = args.device
        self.loss_fn = kwargs['loss_fn']
        self.train_num = 0

    def fit(self, train_loader, optimizer, recorder=None, retain_graph=False):
        self.train().to(self.device)
        description = "Training (the {:d}-batch): tra_Loss = {:.4f}"
        loss_total, avg_loss = 0.0, 0.0
        epochs = tqdm(train_loader, leave=False, desc='local_update')
        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, self.device)
            output = self(batch)
            label = batch['label']
            loss = self.loss_fn(output, label)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            avg_loss = loss_total / (idx + 1)
            epochs.set_description(description.format(idx + 1, avg_loss))
            # recorder['loss'].append(avg_loss)
        # recorder['loss'].append(train_loader.dataset.len)
        self.train_num = len(train_loader.dataset)
        # self.cpu()
        # return loss

    def to_vector(self):
        vec = parameters_to_vector(self.parameters())
        # 函数中的view(-1)会把模型参数tensor的is_leaf属性由True改成False，但require_grad还是True。
        # 函数中的torch.cat()会开启新的内存空间来存储合并后的张量。
        return vec

    def from_vector(self, vec):
        vector_to_parameters(vec, self.parameters())
        # 函数会把param.data加载为vec.data

    def evaluate(self, test_loader, user_offset=None, recorder=None, ifprint=False):
        try:
            test_loader.dataset.reset()
        except:
            pass
        self.eval().to(self.device)
        # loss_total = 0.0
        label,pred=[],[]
        with torch.no_grad():
            loss_total, avg_loss = 0.0, 0.0
            with tqdm(test_loader) as epochs:
                for idx, batch in enumerate(epochs):
                    batch = to_device(batch, self.device)
                    output = self(batch)
                    loss = self.loss_fn(output, batch['label'])
                    loss_total += loss.item()
                    pred += output.tolist()
                    label+=batch['label'].tolist()
            avg_loss = loss_total / (idx+1)

        rec=['ml-1m','ml-100k']
        classifi = ['femnist', 'fashionmnist', 'cifar10', 'cifar100']
        if any(key in self.args.dataset for key in rec):  
            # pred = torch.sigmoid(torch.tensor(pred)).tolist()
            rec_metrics(pred, label, avg_loss, user_offset, test_loader, recorder, ifprint)
        elif any(key in self.args.dataset for key in classifi):
            classfi_metrics(pred, label, avg_loss, user_offset, test_loader, recorder, ifprint)
        self.cpu()

def eva(pre, ground_truth):
    hit5, recall5, ndcg5, hit10, recall10, ndcg10 = 0, 0, 0, 0, 0, 0
    epsilon = 0.1 ** 10
    for i in range(len(ground_truth)):
        one_DCG5, one_recall5, idcg5, one_hit5, one_DCG10, one_recall10, idcg10, one_hit10 = 0,0,0,0,0,0,0,0
        _,idx=np.unique(pre[i],return_index=True)
        p=pre[i][np.sort(idx)]
        top_5_item = p[0:5].tolist()
        top_10_item = p[0:10].tolist()
        positive_item = ground_truth[i] 
        for pos, iid in enumerate(top_5_item):
            if iid in positive_item:
                one_recall5 += 1
                one_DCG5 += 1/np.log2(pos+2)
        for pos, iid in enumerate(top_10_item):
            if iid in positive_item:
                one_recall10 += 1
                one_DCG10 += 1/np.log2(pos+2)

        num5, num10 = min(5, len(positive_item)), min(10, len(positive_item))
        for pos in range(num5):
            idcg5 += 1/np.log2(pos+2)
        for pos in range(num10):
            idcg10 += 1/np.log2(pos+2)

        ndcg5 += one_DCG5 / max(idcg5, epsilon)
        ndcg10 += one_DCG10 / max(idcg10, epsilon)
        top_5_item = set(top_5_item)
        top_10_item = set(top_10_item)
        positive_item = set(positive_item)
        if len(top_5_item & positive_item) > 0:
            hit5 += 1
        if len(top_10_item & positive_item) > 0:
            hit10 += 1
        recall5 += len(top_5_item & positive_item) / max(len(positive_item), epsilon)
        recall10 += len(top_10_item & positive_item) / max(len(positive_item), epsilon)
        #F1 += 2 * precision * recall / max(precision + recall, epsilon)

    return hit5/len(pre), recall5/len(pre), ndcg5/len(pre), hit10/len(pre), recall10/len(pre), ndcg10/len(pre)

def zero_weights(model):
    for n, p in model.named_parameters():
        p.data.zero_()


def rec_metrics(pred, label, avg_loss, user_offset, test_loader, recorder, ifprint):
    fpr, tpr, thresholds = roc_curve(label, pred)
    auc_score = auc(fpr, tpr)
    if recorder is not None:
        recorder['auc'] += [auc_score]
        recorder['loss'] += [avg_loss]
        if user_offset is not None:
            pred = np.hsplit(np.array(pred), user_offset[1:-1])
            pre, ground_truth = [],[]
            for i in range(len(user_offset[:-1])):
                begin, end = user_offset[i], user_offset[i+1]
                s = np.array(pred[i])                    
                ind = np.argsort(-s, kind='heapsort') 
                '''heapsort is for aviod 0-first results for same pred'''                    
                cand_item = 'cand_item_id'
                pre.append(test_loader.dataset.data[cand_item][begin:end][ind])
                pos = test_loader.dataset[begin:end]['label'] == 1
                ground_truth.append(test_loader.dataset.data[cand_item][begin:end][pos])
            hit5, recall5, ndcg5, hit10, recall10, ndcg10=eva(pre, ground_truth)
            recorder['hit5'] += [hit5]
            recorder['recall5'] += [recall5]
            recorder['ndcg5'] += [ndcg5]
            recorder['hit10'] += [hit10]
            recorder['recall10'] += [recall10]
            recorder['ndcg10'] += [ndcg10]
            # recorder['loss'] += [avg_loss]
    if ifprint:
        print('auc: ', auc_score)
        print('loss: ', avg_loss)

def classfi_metrics(pred, label, avg_loss, user_offset, test_loader, recorder, ifprint):
    with torch.no_grad():
        pred=torch.tensor(pred)
        label=torch.tensor(label)
        pred = pred.argmax(dim=1)
        correct=pred.eq(label).sum().item()
        acc = correct/len(test_loader.dataset)
        recorder['acc']+=[acc]
        recorder['loss'] += [avg_loss]
    if ifprint:
        print('acc: ', acc)
        print('loss: ', avg_loss)

def kl_loss(args):
    if args.dataset == 'femnist' or args.dataset == 'fashionmnist':
        loss_fn = torch.nn.KLDivLoss('batchmean')
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-100k':
        loss_fn = binary_kl
    return loss_fn

def binary_kl(input,target):
    input=torch.exp(input)
    loss = input*torch.log(input/target)+(1-input)*torch.log((1-input)/(1-target))
    loss=torch.mean(loss)
    return loss
