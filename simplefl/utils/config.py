"""
Configuration management module for Simple-FL
"""

from jsonargparse import ArgumentParser
import yaml
import torch
from torch.utils.data import Sampler, DataLoader
import pandas as pd
import numpy as np


def init_args():
    """
    Initialize and parse arguments from config files and command line
    
    Returns:
        Parsed arguments object
    """
    parser = ArgumentParser(default_config_files=["./configs/config.yaml"])
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--theme", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--pick_ratio", type=str, default="")
    parser.add_argument(
        "--dataset", type=str, default="ml-1m", help="mcc, ml-1m, alipay"
    )
    parser.add_argument("--datatype", type=str, default="h5", help="pkl, csv")
    parser.add_argument(
        "--datasize", type=str, default="demo", help="size: {demo, full}"
    )
    parser.add_argument("--local_batch_size", type=int, default=20)
    parser.add_argument("--server_batch_size", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=20)
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device: {cuda, cpu}"
    )
    parser.add_argument(
        "--method", type=str, default="fedavg", help="fedavg, fedprox, pba,pga"
    )
    parser.add_argument("--iid", type=str, default="False")
    parser.add_argument(
        "--local_optimizer", type=str, default="sgd", help="{sgd, adam}"
    )
    parser.add_argument(
        "--server_optimizer", type=str, default="sgd", help="{sgd, adam}"
    )
    parser.add_argument("--client_fraction", type=float, default=0.1)
    parser.add_argument("--proxy_ratio", type=float, default=0)
    parser.add_argument("--communication_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--server_epochs", type=int, default=5)
    parser.add_argument("--n_clients", type=int)
    parser.add_argument("--clients_per_round", type=int)
    parser.add_argument("--validate_client_ratio", type=float, default=0)
    parser.add_argument("--avg", type=str, default="w")
    parser.add_argument("--lr_l", type=float, default=1e-2)
    parser.add_argument("--lr_g", type=float, default=1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1, help="prox_coefficient")
    parser.add_argument(
        "--weight-decay", type=float, default=0, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument(
        "--hidden_size", type=list, default=[32, 16], help="hidden_size"
    )
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--features")
    parser.add_argument("--embedding")
    parser.add_argument("--F", type=float, default=1)
    parser.add_argument("--H", type=float, default=1)
    parser.add_argument("--R", type=float, default=1e-5)
    parser.add_argument("--Q", type=float, default=1e-5)
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--FL_validate_clients", type=bool, default=False)
    parser.add_argument("--lora_rank", type=int, help="lora_rank for fedlow")
    parser.add_argument("--cnn_version", type=str, default="2layer", help="CNN_FEMNIST version: {2layer, 3layer}")

    args = parser.parse_args()

    # Track command line arguments
    cmd_args = {}
    for arg in parser.args:
        key, _ = arg.split("=")
        cmd_args[key.lstrip("--")] = args[key.lstrip("--")]

    # Load dataset-specific config
    with open("./configs/datasets.yaml") as f:
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
        for k in dataconfig.keys():
            if k not in cmd_args:
                if k in args.dataset:
                    for k, v in dataconfig[k].items():
                        setattr(args, k, v)

    # Load method-specific config
    with open("./configs/methods.yaml") as f:
        methodconfig = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in methodconfig[args.dataset][args.method].items():
            if k not in cmd_args:
                setattr(args, k, v)

    args.use_feats = {vs: k for k, v in args.features.items() for vs in v}
    
    # Initialize recorder for metrics
    args.recorder = {
        "loss": [],
        "hit5": [],
        "recall5": [],
        "ndcg5": [],
        "auc": [],
        "hit10": [],
        "recall10": [],
        "ndcg10": [],
    }

    return args


class Dataset:
    """
    Dataset wrapper class
    """
    def __init__(self, data, args):
        self.args = args
        self.data = {}
        self.raw = data
        for k in args.use_feats.keys():
            self.data[k] = data[k]
        self.len = len(data)

    def __getitem__(self, idx):
        x = {}
        for key, value in self.data.items():
            x[key] = value[idx]
        return x

    def __len__(self):
        return self.len


class CSV_Dataset:
    """
    CSV Dataset loader for streaming large datasets
    """
    def __init__(self, file, args):
        self.args = args
        self.file = file
        self.cols = list(args.use_feats.keys())
        self.convert = {
            key: lambda x: list(map(int, x.split(",")))
            for key in args.use_feats.keys()
            if "seq" in key
        }

    def reset(self):
        self.data = pd.read_csv(
            self.file, usecols=self.cols, converters=self.convert, iterator=True
        )

    def __getitem__(self, batchsize):
        x = self.data.get_chunk(batchsize)
        x = x.to_dict(orient="list")
        for k in self.args.use_feats.keys():
            x[k] = torch.tensor(x[k])
        return x

    def __len__(self):
        if self.args.datasize == "demo":
            return 10000
        elif "train" in self.file:
            return 9526571
        elif "test" in self.file:
            return 3555419


def CSV_DataLoader(dataset, batchsize=None):
    """Create DataLoader for CSV dataset"""
    sampler = CSV_Sampler(dataset)
    batch_sampler = CSV_Batch_Sampler(sampler, batchsize)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x[0])


class CSV_Sampler(Sampler):
    """Sampler for CSV dataset"""
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return self.data_source.data

    def __len__(self):
        return len(self.data_source)


class CSV_Batch_Sampler(Sampler):
    """Batch sampler for CSV dataset"""
    def __init__(self, csv_sampler, batchsize):
        self.sampler = csv_sampler
        self.batch_size = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return [self.batch_size]

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
