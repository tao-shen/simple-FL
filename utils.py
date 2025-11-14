from ast import arguments
from jsonargparse import ArgumentParser
from torch.utils.data import Sampler, DataLoader
import torch
import yaml
import h5py
import json
import pymysql
import sqlite3
import time
import pandas as pd
import numpy as np
import wandb


def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def init_args():
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

    cmd_args = {}
    for arg in parser.args:
        key, _ = arg.split("=")
        cmd_args[key.lstrip("--")] = args[key.lstrip("--")]

    with open("./configs/datasets.yaml") as f:
        dataconfig = yaml.load(f, Loader=yaml.FullLoader)
        for k in dataconfig.keys():
            if k not in cmd_args:
                if k in args.dataset:
                    for k, v in dataconfig[k].items():
                        setattr(args, k, v)

    with open("./configs/methods.yaml") as f:
        methodconfig = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in methodconfig[args.dataset][args.method].items():
            if k not in cmd_args:
                setattr(args, k, v)

    args.use_feats = {vs: k for k, v in args.features.items() for vs in v}
    # record loss & auc
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
    sampler = CSV_Sampler(dataset)
    batch_sampler = CSV_Batch_Sampler(sampler, batchsize)
    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x[0])


class CSV_Sampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return self.data_source.data

    def __len__(self):
        return len(self.data_source)


class CSV_Batch_Sampler(Sampler):
    def __init__(self, csv_sampler, batchsize):
        self.sampler = csv_sampler
        self.batch_size = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        return [self.batch_size]

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def create_table(args, table_name):
    # generate sql
    names = ""
    for k in args.as_dict().keys():
        if k == "recorder":
            for rk in args.recorder.keys():
                names += rk + " VARCHAR(255), "
            names += "recorder" + " TEXT, "
        else:
            names += k + " VARCHAR(255), "

    sql1 = (
        "CREATE TABLE IF NOT EXISTS {} (exp_id INT UNSIGNED AUTO_INCREMENT, ".format(
            table_name
        )
        + names.replace("TEXT", "LONGTEXT")
        + "submission_date DATETIME, PRIMARY KEY(exp_id));"
    )
    sql2 = (
        "CREATE TABLE IF NOT EXISTS {} (exp_id INTEGER PRIMARY KEY AUTOINCREMENT, ".format(
            table_name
        )
        + names
        + "submission_date DATETIME);"
    )
    # execute sql
    try:
        sql_execute(sql2, "local")
    except:
        pass
    sql_execute(sql1, "remote")


def sql_add_column():
    table = "FL"
    sql = "ALTER TABLE {} ADD acc VARCHAR(255) AFTER ndcg10".format(table)
    sql = "ALTER TABLE {} ADD acc VARCHAR(255)".format(table)


def sql_insert(args, table_name):
    # generate sql
    fields = " ( "
    values = " VALUES ("
    for k, v in args.items():
        if k == "recorder":
            for rk, rv in v.items():
                fields += str(rk) + ", "
                mean = np.mean(rv[-100:])
                std = np.std(rv[-100:])
                ms = {"mean": mean, "std": std}
                values += "'" + str(ms).replace("'", '"') + "'" + ", "
        if isinstance(v, dict):
            fields += str(k) + ", "
            values += "'" + str(v).replace("'", '"') + "'" + ", "
            json.dumps(v)
        elif isinstance(v, np.ndarray):
            fields += str(k) + ", "
            values += "'" + str(v).replace("'", '"') + "'" + ", "
        else:
            fields += str(k) + ", "
            values += "'" + str(v) + "'" + ", "
    fields += "submission_date)"
    values += "'" + time.strftime("%Y-%m-%d %H:%M:%S") + "'"
    sql = "INSERT INTO " + table_name + fields + values + ");"
    # execute sql
    try:
        sql_execute(sql, "local")
    except:
        pass
    sql_execute(sql, "remote")


def save_results(args, table_name):
    if "-" in table_name:
        table_name = table_name.replace("-", "_")
    create_table(args, table_name)
    sql_insert(args, table_name)


def sql_execute(sql, loc=None):
    if loc == "remote":
        con1 = pymysql.connect(
            host="10.72.74.136",
            port=13306,
            user="st",
            passwd="st2318822",
            db="results",
        )
        with con1:
            with con1.cursor(pymysql.cursors.DictCursor) as cur1:
                cur1.execute(sql)
                con1.commit()
                results = cur1.fetchall()
    elif loc == "local":

        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        con2 = sqlite3.connect("results.db")
        con2.row_factory = dict_factory
        cur2 = con2.cursor()
        with con2:
            cur2.execute(sql)
            con2.commit()
            results = cur2.fetchall()
            cur2.close()
    return results


def select(sql):
    sql = "select * from centralized where dataset like 'ml-1m%'"
    sql_execute(sql, "remote")


class Container:
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
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = to_device(x[i], device)
    elif isinstance(x, Container):
        x = to_device(x.data, device)
    # elif isinstance(x, Container_mcc):
    #     x = to_device(x.data, device)
    else:
        x = x.to(device)
    return x

