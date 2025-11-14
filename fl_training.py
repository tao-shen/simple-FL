from server_client import *
from methods import *
from utils import *
from data import *

# args init
args = init_args()
setup_seed(args.seed)
# args.theme = "iclr2025_baselines"

#! FOR FEDLEO
# args.lr_l = 0.01
args.FL_validate_clients = True
args.validate_client_ratio = 0.1

#! FOR FEDLEO

exp_name = "_".join([args.theme, args.dataset])
try:
    mode = "disabled"
    wandb.init(project=exp_name, name=args.method, config=args)
except:
    print("No Wandb!")

# dataset init
init = Data_init(args)

# server/clients init
server = Server(init, args)
clients = init_clients(init, args)

# FL init
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
    if args.wandb:
        wandb.log({k: v[-1] for k, v in fl.args.recorder.items()})

if args.method == "fedleo" and args.FL_validate_clients == True:
    torch.save(
        fl.agg_opt,
        "./assets/"
        + "_".join(["aggr", args.dataset, args.iid, str(args.seed)])
        + ".pt",
    )
save_results(args, exp_name)
# torch.save(fl.aggr, "_".join(["aggr", args.dataset, args.iid]) + ".pt")
