"""
Centralized Training Script for Simple-FL

This script runs centralized (non-federated) training as a baseline.
"""

import torch
import h5py
import json
from torch.utils.data import DataLoader

# Import from simplefl package
from simplefl.core import Data_init
from simplefl.utils import init_args, setup_seed, save_results, Dataset

# Initialize hyperparameters
args = init_args()

# Set random seed
setup_seed(args.seed)

# Initialize data
init = Data_init(args, only_digits=False)

# Load dataset
train_data, test_data = init.data
# train_data=init.proxy_data

# Initialize model
# model = din(args)
model = init.model

# Remove user-specific features for centralized training
# train_data = np.random.choice(train_data, int(0.01*len(train_data)))
for k in train_data.dtype.names:
    if 'user' in k:
        train_data[k] = 0

# Create datasets and dataloaders
train_set = Dataset(train_data, args)
test_set = Dataset(test_data, args)
train_loader = DataLoader(train_set, batch_size=args.local_batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.local_batch_size)

# args.weight_decay=1e-3

# Initialize optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=args.weight_decay)

if __name__ == '__main__':
    # Start training & testing
    if args.communication_rounds == None:
        args.communication_rounds = int(1000)
    
    # Initial evaluation
    model.evaluate(test_loader, user_offset=init.user_offset['test'], recorder=args.recorder, ifprint=True)
    # print(args.recorder['auc'][-1])
    
    # Training loop
    for r in range(args.communication_rounds):
        print('==========the {}-th round==========='.format(r))
        model.fit(train_loader, optimizer)
        model.evaluate(test_loader, user_offset=init.user_offset['test'], recorder=args.recorder, ifprint=True)
        # print(args.recorder['auc'][-1])
    
    # Save model (optional)
    # torch.save(model, 'ml-1m-proxy_0.01.pt')
    
    # Save results
    save_results(args, 'centralized')
