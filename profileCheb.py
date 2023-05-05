import sparsifier
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

import faulthandler
faulthandler.enable()

# import pyRAPL
# pyRAPL.setup()

# take first argument as dataset name
dataset_name = sys.argv[1]
print("dataset_name: ", dataset_name)
# dataset_name = "ogbn_proteins" # ogbn_products, ogbn_proteins
prune_algo = "er_max" # sym_random, sym_degree, er

# ------------------------------------

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)


dataset = PygNodePropPredDataset(
    root=osp.join(PROJECT_HOME, "data"),
    name=dataset_name.replace('_', '-'),
    transform=T.ToSparseTensor(attr="edge_attr"),
)
test_data = dataset[0]

# Move edge features to node features. Only for ogbn-proteins
if dataset_name == "ogbn_proteins":
    test_data.x = test_data.adj_t.mean(dim=1)
    test_data.adj_t.set_value_(None)
    test_data.adj_t = test_data.adj_t.to_tensor()

# # Pre-compute GCN normalization.
# adj_t = test_data.adj_t.set_diag()
# deg = adj_t.sum(dim=1).to(torch.float)
# deg_inv_sqrt = deg.pow(-0.5)
# deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
# adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
# test_data.adj_t = adj_t

pools = [ThreadPool(1) for _ in range(4)]  # 1 pool for each GPU
results = []

dir_list = os.listdir(f"./data/{dataset_name}/pruned/{prune_algo}/")
dir_list = [x for x in dir_list if float(x) > 0.95]
# dir_list = ["0.96"]
print(f"prune rates: {dir_list}")
for i, prune_rate in enumerate(dir_list):
    # load pruned data
    train_data = copy.deepcopy(test_data)
    pruned_el = torch.tensor(
        np.transpose(
            np.loadtxt(
                osp.join(
                    PROJECT_HOME,
                    # f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/duw.el",
                    f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/dw.wel",
                ),
                dtype=float,
            )
        )
    )
    # print(pruned_el.shape)
    # print(pruned_el)
    # adj_t = SparseTensor.from_edge_index(
    #     pruned_el.to(torch.long), sparse_sizes=test_data.adj_t.sparse_sizes()
    # )
    # adj_t.set_value_(pruned_el[2].to(torch.float))

    adj_t = pruned_el[0:2].to(torch.long)
    train_data.adj_t = adj_t
    # print(adj_t)

    edge_weight = pruned_el[2].to(torch.float)
    train_data.edge_weight = edge_weight
    # print(edge_weight)

    # train_data.adj_t = pruned_el.to(torch.long)

    # exit()
    # Pre-compute GCN normalization.
    # set diagonal of adj_t to mean of edge weights
    
    # adj_t = adj_t.set_diag(adj_t.mean())
    # print(adj_t)
    # deg = adj_t.sum(dim=1).to(torch.float)
    # deg_inv_sqrt = deg.pow(-0.5)
    # deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    # print(adj_t)
    # # normalize edge weights in exp way
    # csr = adj_t.to_torch_sparse_csr_tensor()
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
    # train_data.adj_t = adj_t

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=dataset_name.replace('_', '-'))

    if dataset_name == "ogbn_products":
        if float(prune_rate) > 0.8:
            results.append(
                pools[i % len(pools)].apply_async(
                    workload.Cheb_products,
                    (train_data, train_data, split_idx, evaluator, dataset_name),
                    {"experiment_dir_postfix": f"{prune_algo}/{prune_rate}", 
                    "epochs": 1000,
                    "eval_each": 10,
                    "gpu_id": i % len(pools),
                    "use_sage": False},
                )
            )
        else:
            results.append(
                pools[i % len(pools)].apply_async(
                    workload.Cheb_products,
                    (train_data, train_data, split_idx, evaluator, dataset_name),
                    {"experiment_dir_postfix": f"{prune_algo}/{prune_rate}", 
                    "epochs": 500, 
                    "eval_each": 10,
                    "gpu_id": i % len(pools),
                    "use_sage": False},
                )
            )
    if dataset_name == "ogbn_proteins":
        if float(prune_rate) > 0.8:
            results.append(
                pools[i % len(pools)].apply_async(
                    workload.Cheb_products,
                    (train_data, train_data, split_idx, evaluator, dataset_name),
                    {"experiment_dir_postfix": f"{prune_algo}/{prune_rate}", 
                    "epochs": 20000, 
                    "gpu_id": i % len(pools),
                    "use_sage": False},
                )
            )
        else:
            results.append(
                pools[i % len(pools)].apply_async(
                    workload.Cheb_products,
                    (train_data, train_data, split_idx, evaluator, dataset_name),
                    {"experiment_dir_postfix": f"{prune_algo}/{prune_rate}", 
                    "epochs": 5000, 
                    "gpu_id": i % len(pools),
                    "use_sage": False},
                )
            )


for pool in pools:
    pool.close()
    pool.join()

for result in results:
    result.get()



# ------------------------------------


# # reddit_train = dataLoader.Reddit()
# reddit_test = dataLoader.Reddit()

# # loaded pruned graph for training graph
# reddit_train = reddit_test.copy()
# reddit_train.data["edge_index"] = torch.tensor([[], []], dtype=torch.long)

# print(reddit_train.data)
# print(reddit_test.data)

# workload.ClusterGCN(reddit_train, reddit_train, "Reddit")

# # list all dir in ./data/Reddit/pruned/random/
# dir_list = os.listdir("./data/Reddit/pruned/in_degree/")

# for prune_rate in dir_list:
#     # for prune_rate in ["0.999"]:
#     print(prune_rate)
#     pruned_el = torch.tensor(
#         np.transpose(
#             np.loadtxt(
#                 f"./data/Reddit/pruned/in_degree/{prune_rate}/uduw.el", dtype=np.int64
#             )
#         )
#     )
#     adj = SparseTensor.from_edge_index(pruned_el)
#     adj = adj.to_symmetric()
#     row, col, edge_attr = adj.t().coo()
#     edge_index = torch.stack([row, col], dim=0)
#     reddit_train.data["edge_index"] = edge_index
