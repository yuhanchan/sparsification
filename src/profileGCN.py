import dataLoader
import workload
import numpy as np
import torch
from torch_sparse import SparseTensor
import os
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from multiprocessing.pool import ThreadPool
import multiprocessing
import torch_geometric.transforms as T
from torch_geometric.typing import OptTensor
import os.path as osp
import copy
import sys
import argparse

import faulthandler
faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='ogbn-proteins')
parser.add_argument('--prune_algo', type=str, required=True, help="original: train and test on original graph\n empty: train on empty graph, test on full graph\n [prune_algo]: train on pruned graph, test on full graph")
parser.add_argument('--gpu_id', type=int, required=True, default=0)
parser.add_argument("--sage", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)

dataset = PygNodePropPredDataset(
    args.dataset_name,
    root=osp.join(PROJECT_HOME, "data"),
    transform=T.ToSparseTensor(attr="edge_attr"),
)
test_data = dataset[0]

# Move edge features to node features. Only for ogbn-proteins
if args.dataset_name == "ogbn-proteins":
    test_data.x = test_data.adj_t.mean(dim=1)
    test_data.adj_t.set_value_(None)

# Pre-compute GCN normalization.
adj_t = test_data.adj_t.set_diag()
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
test_data.adj_t = adj_t

# train and test on original data
if args.prune_algo == "original":
    train_data = copy.deepcopy(test_data)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=args.dataset_name.replace('_', '-'))
    workload.GCN_proteins(train_data, train_data, split_idx, evaluator, args.dataset_name,
                            experiment_dir_postfix=f"original", 
                            epochs=5000, gpu_id=args.gpu_id, use_sage=args.sage)

# train on empty graph, test on full graph
elif args.prune_algo == "empty": 
    adj_t = SparseTensor(row=torch.tensor([], dtype=torch.long), col=torch.tensor([], dtype=torch.long), value=torch.tensor([]), sparse_sizes=(test_data.num_nodes, test_data.num_nodes))
    adj_t = adj_t.set_diag(1) # add only self loops
    train_data.adj_t = adj_t
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=args.dataset_name.replace('_', '-'))
    workload.GCN_proteins(train_data, train_data, split_idx, evaluator, args.dataset_name,
                            experiment_dir_postfix=f"empty", 
                            epochs=5000, gpu_id=args.gpu_id, use_sage=args.sage)

# train on pruned graph, test on full grpah
else:
    prune_rates = os.listdir(osp.join(PROJECT_HOME, f"data/{args.dataset_name}/pruned/{args.prune_algo}/"))
    print(f"prune rates: {prune_rates}")
    for i, prune_rate in enumerate(prune_rates):
        print(f"dataset: {args.dataset_name}, prune algo: {args.prune_algo}, prune rate: {prune_rate}")
        if not osp.exists(osp.join(PROJECT_HOME, f"data/{args.dataset_name}/pruned/{args.prune_algo}/{prune_rate}/0/duw.el")):
            continue

        # load pruned data
        train_data = copy.deepcopy(test_data)
        pruned_el = torch.tensor(
            np.transpose(
                np.loadtxt(
                    osp.join(
                        PROJECT_HOME,
                        f"data/{args.dataset_name}/pruned/{args.prune_algo}/{prune_rate}/0/duw.el",
                        # f"data/{args.dataset_name}/pruned/{args.prune_algo}/{prune_rate}/0/dw.wel",
                    ),
                    dtype=float,
                )
            )
        )
        adj_t = SparseTensor.from_edge_index(
            pruned_el.to(torch.long), sparse_sizes=test_data.adj_t.sparse_sizes()
        )
        
        adj_t = adj_t.set_diag(adj_t.mean())
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        csr = adj_t.to_torch_sparse_csr_tensor()

        train_data.adj_t = adj_t

        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=args.dataset_name.replace('_', '-'))

        if float(prune_rate) > 0.8: # train more epochs for more pruned graphs
            workload.GCN_proteins(train_data, train_data, split_idx, evaluator, args.dataset_name,
                                experiment_dir_postfix=f"{args.prune_algo}/{prune_rate}", 
                                epochs=20000, gpu_id=args.gpu_id, use_sage=args.sage)
        else:
            workload.GCN_proteins(train_data, train_data, split_idx, evaluator, args.dataset_name,
                                experiment_dir_postfix=f"{args.prune_algo}/{prune_rate}", 
                                epochs=5000, gpu_id=args.gpu_id, use_sage=args.sage)



# if args.dataset_name == "ogbn-products":
#     if float(prune_rate) > 0.8:
#         results.append(
#             pools[i % len(pools)].apply_async(
#                 workload.GCN_products,
#                 (train_data, train_data, split_idx, evaluator, args.dataset_name),
#                 {"experiment_dir_postfix": f"{args.prune_algo}/{prune_rate}", 
#                 "epochs": 2000,
#                 "eval_each": 10,
#                 "gpu_id": i % len(pools),
#                 "use_sage": False},
#             )
#         )
#     else:
#         results.append(
#             pools[i % len(pools)].apply_async(
#                 workload.GCN_products,
#                 (train_data, train_data, split_idx, evaluator, args.dataset_name),
#                 {"experiment_dir_postfix": f"{args.prune_algo}/{prune_rate}", 
#                 "epochs": 1000, 
#                 "eval_each": 10,
#                 "gpu_id": i % len(pools),
#                 "use_sage": False},
#             )
#         )


# # normalize edge weights in exp way
# for row in range(csr.crow_indices().shape[0] - 1):
#     if not row % 10000:
#         print(f"row: {row}")
#     exp_sum = 0
#     for col in range(csr.crow_indices()[row], csr.crow_indices()[row + 1]):
#         csr.values()[col] = np.exp(csr.values()[col])
#         exp_sum += csr.values()[col]
#     for col in range(csr.crow_indices()[row], csr.crow_indices()[row + 1]):
#         csr.values()[col] /= exp_sum
# adj_t = SparseTensor.from_torch_sparse_csr_tensor(csr)
# print(adj_t)