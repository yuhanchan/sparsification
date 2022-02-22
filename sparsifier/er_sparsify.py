import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, csc_matrix
from torch_geometric.transforms import ToSparseTensor
from scipy import sparse, stats
from numpy import inf

from torch_geometric.utils import get_laplacian, from_scipy_sparse_matrix
from torch_sparse import SparseTensor
import myLogger

npz_file_path = None
csv_file_path = None
pkl_file_path = None
prune_file_dir = None

def compute_reff(W, V):
    print("Started computing effective resistances.")
    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))
    n = np.shape(W)[0]
    Reff = sparse.lil_matrix((n, n))
    for orig, end in zip(start_nodes, end_nodes):
        Reff[orig, end] = np.linalg.norm(V[orig, :] - V[end, :]) ** 2
    print("Finished computing effective resistances!")
    return Reff


def stage1(dataset):
    """
    Stage 1: Read the Dataset from pytorch geometric and write it into an .npz file
    Input:
        dataset: PygDataset
    """
    if not osp.exists(npz_file_path):
        sparse_transform = ToSparseTensor()
        sparse_t_data = sparse_transform(dataset.data)
        scipy_data = sparse_t_data.adj_t.to_scipy(layout="csc")
        sparse.save_npz(npz_file_path, scipy_data)
        myLogger.info(message=f"Generated the npz file. Now run compute_V.jl.")


def stage2():
    """
    Stage 2: Run the Julia script that loads the npz file and generates the V matrix
    """
    if not os.exists(csv_file_path):
        cwd = os.getcwd()
        current_file_dir = osp.dirname(osp.realpath(__file__))
        os.chdir(current_file_dir)
        os.system(f"julia compute_V.jl --filepath={npz_file_path}")
        os.chdir(cwd)

def stage3(dataset):
    V_frame = pd.read_csv(csv_file_path, header=None)
    myLogger.info(message=f"Loaded V.csv")
    V = V_frame.to_numpy()
    N = V.shape[0]
    
    L_edge_idx, L_edge_attr = get_laplacian(dataset.data.edge_index)
    L = SparseTensor(row=L_edge_idx[0], col=L_edge_idx[1], value=L_edge_attr)
    L_scipy = L.to_scipy(layout="csc")
    L_diag = csc_matrix(
        (L_scipy.diagonal(), (np.arange(N), np.arange(N))), shape=(N, N)
    )
    W_scipy = L_diag - L_scipy
    if not osp.exists(pkl_file_path):
        R_eff = compute_reff(W_scipy, V)
        with open(pkl_file_path, "wb") as f:
            pickle.dump(R_eff, f)
            print( f"Saved effective resistances in Reff.pkl for future use.")
            f.close()
    else:
        with open(pkl_file_path, "rb") as f:
            R_eff = pickle.load(f)

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W_scipy))

    weights = np.maximum(0, weights)
    Re = np.maximum(0, R_eff[start_nodes, end_nodes].toarray())
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)
    Pe = np.squeeze(Pe)

    # Rudelson, 1996 Random Vectors in the Isotropic Position
    # (too hard to figure out actual C0)
    C0 = 1 / 30.0

    # Rudelson and Vershynin, 2007, Thm. 3.1
    C = 4 * C0

    # 0.094 -> 0.1
    # 0.122 -> 0.2
    # 0.148 -> 0.3
    # 0.176 -> 0.4
    # 0.209 -> 0.5
    # 0.25 -> 0.6
    # 0.306 -> 0.7
    # 0.395 -> 0.8
    # 0.594 -> 0.9
    # 0.87 -> 0.95
    epsilons = [0.094, 0.122, 0.148, 0.176, 0.209, 0.25, 0.306, 0.395, 0.594, 0.87]
    for epsilon in epsilons:
        print("\n------------------------------------------------------")
        print(f"Epsilon: {epsilon}")
        q = round(N * np.log(N) * 9 * C ** 2 / (epsilon ** 2))
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
        print("Edges (before sparsification): ", np.count_nonzero(weights))
        print("Edges (after sparsification): ", np.count_nonzero(new_weights))
        print("Prune rate: ", 1 - np.count_nonzero(new_weights) / np.size(new_weights))
        # convert into PyG's object
        edge_index, edge_weight = from_scipy_sparse_matrix(sparserW)
        torch.save(
            {"edge_index": edge_index, "edge_weight": edge_weight},
            osp.join(prune_file_dir, epsilon, "edge_data.pt")
            )
        
    return  edge_index, edge_weight

def er_sparsify(dataset, dataset_name, epsilon):
    """
    Input:
        dataset: PygDataset
        dataset_name: str, name of the dataset
        epsilon: float, sparsification parameter
    Output:
        data
    """
    global npz_file_path, csv_file_path, pkl_file_path, prune_file_dir
    npz_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/raw/V.npz')
    csv_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/raw/V.csv')
    pkl_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/raw/Reff.pkl')
    prune_file_dir = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/er')
    os.makedirs(prune_file_dir, exist_ok=True)
   
    if osp.exists(ops.join(prune_file_dir, epsilon, "edge_data.pt")):
        edge_data = torch.load(ops.join(prune_file_dir, epsilon, "edge_data.pt"))
        edge_index = edge_data["edge_index"]
        edge_weight = edge_data["edge_weight"]
    else:
        stage1(dataset)
        stage2()
        stage3(dataset)
    data = dataset.data
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    return data