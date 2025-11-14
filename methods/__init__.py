import os
import importlib
import sys
# from .fl import *

sys.path.append("..")

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get all Python files in the current directory,f[:-3]取文件名，去掉.py
method_files = [
    f.replace(".py", "")
    for f in os.listdir(current_dir)
    if f.endswith(".py") and f != "__init__.py"
]

# module = {
#     method: importlib.import_module(f".{method}", package=__name__)
#     for method in method_files
# }
methods = {}
for method in method_files:
    module = importlib.import_module(f".{method}", package=__name__)
    attrs = {attr.lower(): getattr(module, attr) for attr in dir(module)}
    if method not in attrs:
        pass
    else:
        methods[method] = attrs[method]


def fl_methods(server, clients, args):
    return methods[args.method](server, clients, args)


# from .fl import *
# from .fedavg import FedAvg
# from .fedavgm import FedAvgM
# from .gaussian import Gaussian
# from .fedprox import FedProx
# from .fedopt import FedOpt
# from .scaffold import Scaffold
# from .feddf import FedDF
# from .fedmeta import FedMeta
# from .fedleo import FedLeo
# from .fedeve import FedEve
# from .fedavg_client_drift_only import FedAvg_Client_Drift_Only
# from .fedavg_period_drift_only import FedAvg_Period_Drift_Only
# import sys
# sys.path.append("..")


# def fl_methods(server, clients, args):
#     methods = {
#         "fedavg": FedAvg,
#         "gaussian": Gaussian,
#         "fedavg_client_drift_only": FedAvg_Client_Drift_Only,
#         "fedavg_period_drift_only": FedAvg_Period_Drift_Only,
#         "fedavgm": FedAvgM,
#         "fedprox": FedProx,
#         "fedopt": FedOpt,
#         "scaffold": Scaffold,
#         "fedeve": FedEve,
#         "feddf": FedDF,
#         "fedmeta": FedMeta,
#         "fedleo": FedLeo,
#     }
#     return methods[args.method](server, clients, args)
