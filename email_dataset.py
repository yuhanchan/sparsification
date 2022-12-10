# a temperary file for testing on email dataset

import os
import os.path as osp
import subprocess
from concurrent.futures import ProcessPoolExecutor

# random
def random_prune():
    el_path = "./data/email/raw/final.el"
    for drop_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        output_path = f"./data/email/pruned/random/{drop_rate}/duw.el"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.system(
            f"./sparsifier/bin/prune -f {el_path} -q random -p {drop_rate} -o {output_path}"
        )


def in_degree_prune():
    # in_degree
    el_path = "./data/email/raw/final.el"
    degree_to_drop_rate_dict = {
        3: 0.9,
        6: 0.8,
        10: 0.7,
        14: 0.6,
        19: 0.5,
        25: 0.4,
        33: 0.3,
        44: 0.2,
        66: 0.1,
    }

    for degree_thres, drop_rate in degree_to_drop_rate_dict.items():
        output_path = f"./data/email/pruned/in_degree/{drop_rate}/duw.el"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.system(
            f"./sparsifier/bin/prune -f {el_path} -q in_threshold -x {degree_thres} -o {output_path}"
        )


def out_degree_prune():
    # out_degree
    el_path = "./data/email/raw/final.el"
    degree_to_drop_rate_dict = {
        3: 0.9,
        7: 0.8,
        11: 0.7,
        16: 0.6,
        22: 0.5,
        29: 0.4,
        39: 0.3,
        54: 0.2,
        82: 0.1,
    }

    for degree_thres, drop_rate in degree_to_drop_rate_dict.items():
        output_path = f"./data/email/pruned/out_degree/{drop_rate}/duw.el"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.system(
            f"./sparsifier/bin/prune -f {el_path} -q out_threshold -x {degree_thres} -o {output_path}"
        )


import sys
from time import time
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix
from scipy import sparse, stats
from numpy import inf

from torch_geometric.utils import get_laplacian, from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from concurrent.futures import ProcessPoolExecutor

npz_file_path = None
csv_file_path = None
pkl_file_path = None
stage3_file_path = None
prune_file_path = None
prune_file_dir = None


def compute_reff(W, V, reuse=True):
    print("Stage 3a: computing Reff")
    if reuse and osp.exists(pkl_file_path):
        print(f"Reff already exists, loading...")
        with open(pkl_file_path, "rb") as f:
            R_eff = pickle.load(f)
    else:
        print(f"Reff not exist, computing...")
        t_s = time()
        start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))
        print(f"Sparse.find took {time() - t_s} seconds.")
        n = np.shape(W)[0]

        t_ss = time()
        R_eff = sparse.lil_matrix((n, n))
        for orig, end in zip(start_nodes, end_nodes):
            R_eff[orig, end] = np.linalg.norm(V[orig, :] - V[end, :]) ** 2
        print(f"Computation took {time() - t_ss} seconds.")

        t_ss = time()
        with open(pkl_file_path, "wb") as f:
            pickle.dump(R_eff, f)
            print(f"Reff.pkl saved. Took {time() - t_ss} seconds.")
        t_e = time()
        print(f"Stage 3a took {t_e - t_s} seconds.")
    return R_eff


def stage1(dataset, isPygDataset=False, reuse=True):
    print(f"Stage 1: converting pytorch dataset to npz file")
    if reuse and osp.exists(npz_file_path):
        print(f"npz file already exists. Skipping...")
    else:
        print(f"npz file not exist. Computing...")
        t_s = time()
        scipy_data = sparse.csc_matrix(
            (np.ones(dataset.shape[0], int), (dataset[:, 0], dataset[:, 1]))
        )
        sparse.save_npz(npz_file_path, scipy_data)
        print(f"npz file generated.")
        t_e = time()
        print(f"Stage 1 took {t_e - t_s} seconds.")


def stage2(reuse=True):
    """
    Stage 2: Run the Julia script that loads the npz file and generates the V matrix
    """
    print(f"Stage 2: invoking julia script to generate V.csv matrix (Z in the paper)")
    if reuse and osp.exists(csv_file_path):
        print(f"csv file already exists. Skipping...")
    else:
        print(f"csv file not exist. Computing...")
        t_s = time()
        cwd = os.getcwd()
        current_file_dir = osp.dirname(osp.realpath(__file__))
        os.chdir(current_file_dir)
        os.system(f"julia sparsifier/compute_V.jl --filepath={npz_file_path}")
        os.chdir(cwd)
        t_e = time()
        print(f"Stage 2 took {t_e - t_s} seconds.")


def compute_edge_data(
    epsilon: Union[int, float],
    prune_rate,
    Pe,
    C,
    weights,
    start_nodes,
    end_nodes,
    N,
    dataset_name,
):
    """This function is called from stage3. This function is doing the final sampling based on Reff.
    This function is seperated to be able to use future.concurrent
    """
    if not isinstance(epsilon, int) and not isinstance(epsilon, float):
        myLogger.error(f"epsilon must be one of the type: int, float")
        sys.exit(1)
    print(f"Stage 3b: sampling edges based on Reff")
    t_s = time()
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
    sparserW = sparserW + sparserW.T  # make it symmetric
    t_e = time()
    print(f"Stage 3b took {t_e - t_s} seconds.")
    print(
        f"Prune rate for epsilon={epsilon}: {1 - np.count_nonzero(new_weights) / np.size(new_weights)}, ({np.size(new_weights)} -> {np.count_nonzero(new_weights)})"
    )

    # convert into PyG's object
    edge_index, edge_weight = from_scipy_sparse_matrix(sparserW)
    prune_file_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "edge_data.pt",
    )
    npy_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "dw.npy",
    )
    duw_el_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "duw.wel",
    )
    dw_el_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "dw.wel",
    )
    uduw_el_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "uduw.el",
    )
    udw_el_path = osp.join(
        prune_file_dir,
        str(prune_rate),
        "udw.wel",
    )

    t_s = time()
    os.makedirs(osp.dirname(npy_path), exist_ok=True)
    to_save = torch.cat((edge_index, edge_weight.reshape(1, -1)), 0).numpy().transpose()
    np.save(npy_path, to_save)
    np.savetxt(duw_el_path, to_save, fmt="%d %d %.5f")
    print(f"Saved edge_data.pt in {time() - t_s} seconds")
    return edge_index, edge_weight


def stage3(
    dataset,
    dataset_name,
    epsilon: Union[int, float, list],
    isPygDataset=False,
    max_workers=64,
    reuse=True,
):
    print(f"Stage 3: Pruning edges")
    if reuse and osp.exists(stage3_file_path):
        print("stage3.npz already exists, loading")
        stage3_var = np.load(stage3_file_path)
        Pe = stage3_var["Pe"]
        weights = stage3_var["weights"]
        start_nodes = stage3_var["start_nodes"]
        end_nodes = stage3_var["end_nodes"]
        N = stage3_var["N"]
    else:
        V_frame = pd.read_csv(csv_file_path, header=None)
        V = V_frame.to_numpy()
        N = V.shape[0]

        # print(torch.tensor(np.transpose(dataset)))
        L_edge_idx, L_edge_attr = get_laplacian(torch.LongTensor(np.transpose(dataset)))
        L = SparseTensor(
            row=L_edge_idx[0], col=L_edge_idx[1], value=L_edge_attr
        )  # Lapacian, nxn
        L_scipy = L.to_scipy(layout="csc")
        L_diag = csc_matrix(
            (L_scipy.diagonal(), (np.arange(N), np.arange(N))), shape=(N, N)
        )
        W_scipy = L_diag - L_scipy  # Weight matrix, nxn

        # compute Reff
        R_eff = compute_reff(W_scipy, V, reuse=reuse)

        # only taking the lower triangle of the W_nxn as it is symmetric
        start_nodes, end_nodes, weights = sparse.find(sparse.tril(W_scipy))

        weights = np.maximum(0, weights)  # 1xm
        Re = np.maximum(0, R_eff[start_nodes, end_nodes].toarray())  # 1xm
        Pe = weights * Re  # element-wise multiplication, 1xm
        Pe = Pe / np.sum(Pe)  # normalize, 1xm
        Pe = np.squeeze(Pe)  # 1xm

        np.savez(
            stage3_file_path,
            Pe=Pe,
            weights=weights,
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            N=N,
        )
        print(f"stage3.npz saved")

    # Sampling
    # Rudelson, 1996 Random Vectors in the Isotropic Position
    # (too hard to figure out actual C0)
    C0 = 1 / 30.0
    # Rudelson and Vershynin, 2007, Thm. 3.1
    C = 4 * C0

    if isinstance(epsilon, int) or isinstance(epsilon, float):
        edge_index, edge_weight = compute_edge_data(
            epsilon, 0.1, Pe, C, weights, start_nodes, end_nodes, N, dataset_name
        )
    elif isinstance(epsilon, list):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    compute_edge_data,
                    epsilon_,
                    prune_rate,
                    Pe,
                    C,
                    weights,
                    start_nodes,
                    end_nodes,
                    N,
                    dataset_name,
                ): epsilon_
                for (epsilon_, prune_rate) in epsilon
            }
            for future in futures:
                print(f"Epsilon: {futures[future]}")
                future.result()
        edge_index, edge_weight = None, None
    else:
        print(f"epsilon must be one of the type: int, float, list")
        sys.exit(1)


def python_er_sparsify(
    dataset, dataset_name, epsilon: Union[int, float, list], reuse=True
):
    global npz_file_path, csv_file_path, pkl_file_path, stage3_file_path, prune_file_path, prune_file_dir
    npz_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"data/{dataset_name}/raw/V.npz"
    )
    csv_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"data/{dataset_name}/raw/V.csv"
    )
    pkl_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"data/{dataset_name}/raw/Reff.pkl"
    )
    stage3_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"data/{dataset_name}/raw/stage3.npz"
    )

    prune_file_dir = osp.join(
        osp.dirname(osp.abspath(__file__)), f"data/{dataset_name}/pruned/er"
    )
    os.makedirs(prune_file_dir, exist_ok=True)

    if isinstance(epsilon, int) or isinstance(epsilon, float):
        print(f"python_er_sparsify: epsilon: {epsilon}")
        if reuse and prune_file_path and osp.exists(prune_file_path):
            print(f"edge_data.pt already exists. Loading it...")
            edge_data = torch.load(prune_file_path)
            edge_index = edge_data["edge_index"]
            edge_weight = edge_data["edge_weight"]
        else:
            print(f"edge_data.pt does not exist. Computing it...")
            stage1(dataset.copy(), isPygDataset=False, reuse=reuse)
            stage2(reuse=reuse)
            stage3(dataset, dataset_name, epsilon, isPygDataset=False, reuse=reuse)
    elif isinstance(epsilon, list):
        print(f"python_er_sparsify: epsilon: {epsilon}")
        stage1(dataset.copy(), isPygDataset=False, reuse=reuse)
        stage2(reuse=reuse)
        stage3(dataset, dataset_name, epsilon, isPygDataset=False, reuse=reuse)
    else:
        print(f"epsilon must be one of the type: int, float, list")
        sys.exit(1)


# er
def er_prune():
    el_path = "./data/email/raw/final.el"
    el = np.loadtxt(el_path, dtype=np.int32)
    python_er_sparsify(
        el,
        "email",
        [
            (0.12, 0.1),
            (0.155, 0.2),
            (0.185, 0.3),
            (0.22, 0.4),
            (0.265, 0.5),
            (0.315, 0.6),
            (0.39, 0.7),
            (0.51, 0.8),
            (0.77, 0.9),
        ],
    )


def process():
    for prune_algo in ["random", "in_degree", "out_degree"]:
        for prune_rate in [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
        ]:
            folder = f"data/email/pruned/{prune_algo}/{prune_rate}"
            with open(f"{folder}/duw.comp.el", "w") as outfile:
                subprocess.run(
                    ["./utils/bin/utils", "-i", f"{folder}/duw.el", "-m", "10"],
                    stderr=outfile,
                )  # get the largest component
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"{folder}/duw.comp.el",
                        "-o",
                        f"{folder}/final.el",
                        "-m",
                        "5",
                    ]
                )  # elim disconnected comp
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"data/email/raw/final.el",
                        "-o",
                        f"{folder}/original.el",
                        "-p",
                        f"{folder}/final.el.map",
                        "-m",
                        "13",
                    ]
                )  # apply edge map to the raw/final.el
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"{folder}/original.el",
                        "-o",
                        f"{folder}/original.onebase.el",
                        "-m",
                        "7",
                    ]
                )  # change original el from 0-base to 1-base

    for prune_algo in ["er"]:
        for prune_rate in [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
        ]:
            folder = f"data/email/pruned/{prune_algo}/{prune_rate}"
            with open(f"{folder}/duw.comp.el", "w") as outfile:
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"{folder}/duw.wel",
                        "-o",
                        f"{folder}/duw.el",
                        "-m",
                        "12",
                    ]
                )  # remove weight column
                subprocess.run(
                    ["./utils/bin/utils", "-i", f"{folder}/duw.el", "-m", "10"],
                    stderr=outfile,
                )  # get the largest component
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"{folder}/duw.comp.el",
                        "-o",
                        f"{folder}/final.el",
                        "-m",
                        "5",
                    ]
                )  # elim disconnected comp
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"data/email/raw/final.el",
                        "-o",
                        f"{folder}/original.el",
                        "-p",
                        f"{folder}/final.el.map",
                        "-m",
                        "13",
                    ]
                )  # apply edge map to the raw/final.el
                subprocess.run(
                    [
                        "./utils/bin/utils",
                        "-i",
                        f"{folder}/original.el",
                        "-o",
                        f"{folder}/original.onebase.el",
                        "-m",
                        "7",
                    ]
                )  # change original el from 0-base to 1-base


def func1(folder):
    """
    Helps parallelize process_Reddit
    """
    with open(f"{folder}/duw.comp.el", "w") as outfile:
        subprocess.run(
            ["./utils/bin/utils", "-i", f"{folder}/duw.el", "-m", "10"], stderr=outfile
        )  # get the largest component
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"{folder}/duw.comp.el",
                "-o",
                f"{folder}/final.el",
                "-m",
                "5",
            ]
        )  # elim disconnected comp
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"data/Reddit_CGE/raw/duw.el",
                "-o",
                f"{folder}/original.el",
                "-p",
                f"{folder}/final.el.map",
                "-m",
                "13",
            ]
        )  # apply edge map to the raw/final.el
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"{folder}/original.el",
                "-o",
                f"{folder}/original.onebase.el",
                "-m",
                "7",
            ]
        )  # change original el from 0-base to 1-base


def func2(folder):
    """
    Helps parallelize process_Reddit
    """
    with open(f"{folder}/duw.comp.el", "w") as outfile:
        # subprocess.run(["./utils/bin/utils", "-i", f"{folder}/duw.wel", "-o", f"{folder}/duw.el", "-m", "12"]) # remove weight column
        subprocess.run(
            ["./utils/bin/utils", "-i", f"{folder}/duw.el", "-m", "10"], stderr=outfile
        )  # get the largest component
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"{folder}/duw.comp.el",
                "-o",
                f"{folder}/final.el",
                "-m",
                "5",
            ]
        )  # elim disconnected comp
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"data/Reddit_CGE/raw/duw.el",
                "-o",
                f"{folder}/original.el",
                "-p",
                f"{folder}/final.el.map",
                "-m",
                "13",
            ]
        )  # apply edge map to the raw/final.el
        subprocess.run(
            [
                "./utils/bin/utils",
                "-i",
                f"{folder}/original.el",
                "-o",
                f"{folder}/original.onebase.el",
                "-m",
                "7",
            ]
        )  # change original el from 0-base to 1-base


def process_Reddit():
    with ProcessPoolExecutor(max_workers=128) as executor:
        futures = []
        for prune_algo in ["random", "in_degree", "out_degree"]:
            for prune_rate in os.listdir(f"data/Reddit_CGE/pruned/{prune_algo}"):
                folder = f"data/Reddit_CGE/pruned/{prune_algo}/{prune_rate}"
                futures.append(executor.submit(func1, folder))

        for prune_algo in ["er"]:
            for prune_rate in os.listdir(f"data/Reddit_CGE/pruned/{prune_algo}"):
                folder = f"data/Reddit_CGE/pruned/{prune_algo}/{prune_rate}"
                futures.append(executor.submit(func2, folder))

        for future in futures:
            future.result()


# random_prune()
# in_degree_prune()
# out_degree_prune()
# er_prune()

# process()

process_Reddit()
