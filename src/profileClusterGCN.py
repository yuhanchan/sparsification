import dataLoader
import workload
import numpy as np
import torch
from torch_sparse import SparseTensor
import os
import copy 
from os import path as osp
from multiprocessing.pool import ThreadPool

import faulthandler
faulthandler.enable()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prune_algo', type=str, required=True, help="original: train and test on original graph\n empty: train on empty graph, test on full graph\n [prune_algo]: train on pruned graph, test on full graph")
parser.add_argument('--gpu_id', type=int, required=True, default=0)
args = parser.parse_args()

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)


test_data = dataLoader.Reddit()

# train and test on original graph
if args.prune_algo == "original":
    train_data = copy.deepcopy(test_data)
    workload.ClusterGCN(train_data, test_data, "Reddit", 
                        experiment_dir_postfix=f"original", 
                        epochs=100, eval_each=5, gpu_id=args.gpu_id, use_sage=False)

# train on empty graph, test on full graph
elif args.prune_algo == "empty":
    train_data.data["edge_index"] = torch.tensor([list(range(train_data.data.y.shape[0])), list(range(train_data.data.y.shape[0]))], dtype=torch.long)
    workload.ClusterGCN(train_data, test_data, "Reddit", 
                        experiment_dir_postfix=f"empty", 
                        epochs=100, eval_each=5, gpu_id=args.gpu_id, use_sage=False)

# train on pruned graph, test on full graph
else:
    prune_rates = os.listdir(osp.join(PROJECT_HOME, f"data/Reddit/pruned/{args.prune_algo}/"))
    print(f"prune rates: {prune_rates}")
    for i, prune_rate in enumerate(prune_rates):
        if not osp.exists(osp.join(PROJECT_HOME, f"data/Reddit/pruned/{args.prune_algo}/{prune_rate}/0/duw.el")):
            continue

        # load pruned data
        train_data = copy.deepcopy(test_data)
        pruned_el = torch.tensor(
            np.transpose(
                np.loadtxt(
                    osp.join(PROJECT_HOME, f"data/Reddit/pruned/{args.prune_algo}/{prune_rate}/0/duw.el"), 
                    dtype=np.int64
                )
            )
        )
        adj = SparseTensor.from_edge_index(pruned_el)
        # adj = adj.to_symmetric()
        row, col, edge_attr = adj.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        train_data.data["edge_index"] = edge_index

        workload.ClusterGCN(train_data, test_data, "Reddit", 
                            experiment_dir_postfix=f"{args.prune_algo}/{prune_rate}", 
                            epochs=100, eval_each=5, gpu_id=args.gpu_id, use_sage=False)
