"""
Federated Learning Training Script for Simple-FL

This script runs federated learning experiments with various algorithms.
"""

import torch
try:
    import wandb
except ImportError:
    wandb = None

# Import from simplefl package
from simplefl.core import Server, init_clients, Data_init
from simplefl.methods import fl_methods
from simplefl.utils import init_args, setup_seed, save_results

# Initialize arguments
args = init_args()
setup_seed(args.seed)
# args.theme = "iclr2025_baselines"

#! FOR FEDLEO
# args.lr_l = 0.01
# args.FL_validate_clients = True
# args.validate_client_ratio = 0.1
#! FOR FEDLEO

# Setup experiment name
exp_name = "_".join([args.theme, args.dataset])

# Initialize Wandb (optional)
if wandb is not None:
    try:
        mode = "disabled"
        wandb.init(project=exp_name, name=args.method, config=args)
    except:
        print("No Wandb!")
else:
    print("Wandb not installed, skipping...")

# Dataset initialization
init = Data_init(args)

# Server and clients initialization
server = Server(init, args)
clients = init_clients(init, args)

# FL algorithm initialization
fl = fl_methods(server, clients, args)
fl.server_init()
fl.clients_init()
fl.evaluate()

# fl.load_aggregator('lstm2.pt')

# Start FL training
# fl.T = 1 # for debug
for t in range(fl.T):
    print("==========the {}-th round===========".format(t))
    fl.candidates_sampling()
    fl.clients_update()
    fl.server_update()
    fl.evaluate()
    if args.wandb and wandb is not None:
        wandb.log({k: v[-1] for k, v in fl.args.recorder.items()})

# Save FedLeo aggregator if needed
if args.method == "fedleo" and args.FL_validate_clients == True:
    torch.save(
        fl.agg_opt,
        "./assets/"
        + "_".join(["aggr", args.dataset, args.iid, str(args.seed)])
        + ".pt",
    )

# Save results
save_results(args, exp_name)
# torch.save(fl.aggr, "_".join(["aggr", args.dataset, args.iid]) + ".pt")
