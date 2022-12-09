import numpy as np
import torch
import pickle
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from scipy import sparse, stats
from numpy import inf
from torch_geometric.utils import from_scipy_sparse_matrix

# with open("data/Reddit/raw/Reff.pkl", "rb") as f:
#     Re = pickle.load(f)

# stage3 = np.load("data/Reddit/raw/stage3.npz")
# Pe = stage3["Pe"]
# start_nodes = stage3["start_nodes"]
# end_nodes = stage3["end_nodes"]

# print(f"Re.shape: {Re.shape}")
# print(f"Pe.shape: {Pe.shape}")

# print(f"Re[0:10]: {Re[0:10]}")
# print(f"Pe[0:10]: {Pe[0:10]}")

# print(f"start_nodes.shape: {start_nodes.shape}")
# print(f"end_nodes.shape: {end_nodes.shape}")

# print(f"start_nodes[0:10]: {start_nodes[0:10]}")
# print(f"end_nodes[0:10]: {end_nodes[0:10]}")

# idx = zip(start_nodes[0:10], end_nodes[0:10])
# for s, e in idx:
#     print(Re[s, e])


# dataset = PygNodePropPredDataset(
#     name="ogbn-products",
#     transform=T.ToSparseTensor(),
#     root="data",
# )
# data = dataset[0]

# # print(data.adj_t.coo())

# with open("data/ogbn_products/raw/Reff.pkl", "rb") as f:
#     reff = pickle.load(f)
# # Reff = []
# # get the reff for each edge and add it to the Reff

# coo = data.adj_t.coo()
# rows, cols = coo[0], coo[1]

# lower_triangle = np.tril_indices(data.num_nodes)
# for i in range(len(rows)):
#     row, col = rows[i], cols[i]
#     if row > col:

#         print(row, col, reff[row, col], reff[col, row])

#     # Reff.append(reff[row, col])
# # print(row, col)
# # Reff.append(reff[row, col])


var = np.load("data/ogbn_products/raw/stage3.npz")
Pe = var["Pe"]
weights = var["weights"]
start_nodes = var["start_nodes"]
end_nodes = var["end_nodes"]
N = var["N"]

C0 = 1 / 30.0
C = 4 * C0


def sampling(
    epsilon,
    Pe,
    C,
    weights,
    start_nodes,
    end_nodes,
    N,
):
    q = round(N * np.log(N) * 9 * C**2 / (epsilon**2))
    results = np.random.choice(np.arange(np.shape(Pe)[0]), int(q), p=list(Pe))
    spin_counts = stats.itemfreq(results).astype(int)

    per_spin_weights = weights / (q * Pe)
    per_spin_weights[per_spin_weights == inf] = 0

    counts = np.zeros(np.shape(weights)[0])
    counts[spin_counts[:, 0]] = spin_counts[:, 1]
    new_weights = counts * per_spin_weights

    sparserW = sparse.csc_matrix(
        (np.squeeze(new_weights), (start_nodes, end_nodes)), shape=(N, N)
    )

    sparserW = sparserW + sparserW.T

    print(
        f"Prune rate for epsilon={epsilon}: {1 - np.count_nonzero(new_weights) / np.size(new_weights)}, ({np.size(new_weights)} -> {np.count_nonzero(new_weights)})"
    )
    # convert into PyG's object
    edge_index, edge_weight = from_scipy_sparse_matrix(sparserW)
    print(type(edge_index))
    print(edge_index)

    return edge_index, edge_weight


sampling(0.5, Pe, C, weights, start_nodes, end_nodes, N)
