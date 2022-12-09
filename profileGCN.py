import sparsifier
import dataLoader
import workload
import numpy as np
import torch
from torch_sparse import SparseTensor
import os

import faulthandler

faulthandler.enable()

# reddit_train = dataLoader.Reddit()
reddit_test = dataLoader.Reddit()

# loaded pruned graph for training graph
reddit_train = reddit_test.copy()
reddit_train.data["edge_index"] = torch.tensor([[], []], dtype=torch.long)

print(reddit_train.data)
print(reddit_test.data)

workload.ClusterGCN(reddit_train, reddit_train, "Reddit")

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

#     print(reddit_train.data)
#     print(reddit_test.data)

#     # print(reddit_train.data["train_mask"])
#     # print(reddit_train.data["test_mask"])

#     # reddit_train.data["train_mask"] = torch.ones(
#     # reddit_train.data["train_mask"].shape
#     # ).type(torch.bool)

#     # reddit_test.data["test_mask"] = torch.ones(reddit_test.data["test_mask"].shape).type(
#     # torch.bool
#     # )

#     workload.ClusterGCN(reddit_train, reddit_test, "Reddit")
