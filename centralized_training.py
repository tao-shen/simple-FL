from utils import *
from models import *
from data import *
import torch, h5py, json

# initialize hyperparameters
args = init_args()
# random seed
setup_seed(args.seed)
init = Data_init(args, only_digits=False)
# load dataset
train_data, test_data = init.data
# train_data=init.proxy_data
# initialize model
# model = din(args)
model = init.model
# train_data = np.random.choice(train_data, int(0.01*len(train_data)))
for k in train_data.dtype.names:
    if 'user' in k:
        train_data[k] = 0
train_set = Dataset(train_data, args)
test_set = Dataset(test_data, args)
train_loader = DataLoader(train_set, batch_size=args.local_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.local_batch_size)
# args.weight_decay=1e-3
# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=args.weight_decay)

if __name__ == '__main__':
    # start training & testing
    if args.communication_rounds == None:
        args.communication_rounds = int(1000)
    model.evaluate(test_loader, user_offset=init.user_offset['test'], recorder=args.recorder,ifprint=True)
    # print(args.recorder['auc'][-1])
    for r in range(args.communication_rounds):
        print('==========the {}-th round==========='.format(r))
        model.fit(train_loader, optimizer)
        model.evaluate(test_loader, user_offset=init.user_offset['test'], recorder=args.recorder,ifprint=True)
        # print(args.recorder['auc'][-1])
    # torch.save(model, 'ml-1m-proxy_0.01.pt')
    save_results(args, 'centralized')
