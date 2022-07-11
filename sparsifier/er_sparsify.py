import os
import os.path as osp
import sys
from time import time
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix
from torch_geometric.transforms import ToSparseTensor
from scipy import sparse, stats
from numpy import inf

from torch_geometric.utils import get_laplacian, from_scipy_sparse_matrix
from torch_sparse import SparseTensor
import myLogger
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, shared_memory
from subprocess import Popen

npz_file_path = None  # convert original edge list to npz file
csv_file_path = None  # The Z matrix in paper
pkl_file_path = None  # The Reff file
stage3_file_path = None  # parameters used in stage3
prune_file_path = None
prune_file_dir = None


def compute_single_reff(i, V_shared, start_nodes, end_nodes):
    """
    Called from compure_reff if parallel is enabled
    """
    t_s = time()
    res = []
    for orig, end in zip(start_nodes, end_nodes):
        res.append(
            (orig, end, np.linalg.norm(V_shared[orig, :] - V_shared[end, :]) ** 2)
        )
    print(f"worker {i} finished in {time() - t_s} seconds.")
    return res


def compute_reff(W, V, parallel=True, reuse=True):
    """
    This uses the V from julia script Compute_V.jl to calculate the Reff
    V here is the Z in paper [Graph sparsification by effective resistances]

    Input:  W: weight matrix
            V: V matrix
            parallel: True if you want to run the computation in parallel
            reuse: True if you want to reuse the V matrix if exists
    """
    myLogger.info("Stage 3a: computing Reff")
    if reuse and osp.exists(pkl_file_path):
        myLogger.info(message=f"Reff already exists, loading...")
        with open(pkl_file_path, "rb") as f:
            R_eff = pickle.load(f)
    else:
        myLogger.info(message=f"Reff not exist, computing...")
        t_s = time()
        start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))
        myLogger.info(message=f"Sparse.find took {time() - t_s} seconds.")
        n = np.shape(W)[0]

        # # save start_nodes, end_nodes, V to txt file
        # with open('start_nodes.txt', "w") as f:
        #     for i in start_nodes:
        #         f.write(f"{i}\n")
        # with open('end_nodes.txt', "w") as f:
        #     for i in end_nodes:
        #         f.write(f"{i}\n")
        # with open('V.txt', "w") as f:
        #     np.savetxt(f, V, delimiter=",")
        # sys.exit()

        if not parallel:
            t_ss = time()
            R_eff = sparse.lil_matrix((n, n))
            for orig, end in zip(start_nodes, end_nodes):
                R_eff[orig, end] = np.linalg.norm(V[orig, :] - V[end, :]) ** 2
            myLogger.info(message=f"Computation took {time() - t_ss} seconds.")

        else:
            # make V and R_eff shared_memory to avoid copying in multiprocessing
            t_ss = time()
            R_eff = sparse.lil_matrix((n, n))

            shm1 = shared_memory.SharedMemory(create=True, size=V.nbytes)
            V_shared = np.ndarray(V.shape, dtype=V.dtype, buffer=shm1.buf)
            V_shared[:] = V[:]

            num_processes = 64
            with Pool(num_processes) as p:
                results = p.starmap(
                    compute_single_reff,
                    [
                        (i, V_shared, start_nodes_part, end_nodes_part)
                        for i, start_nodes_part, end_nodes_part in zip(
                            np.arange(num_processes) + 1,
                            np.array_split(start_nodes, num_processes),
                            np.array_split(end_nodes, num_processes),
                        )
                    ],
                )
            myLogger.info(message=f"Parallel computation took {time() - t_ss} seconds.")
            shm1.close()
            shm1.unlink()

            # collect results
            t_ss = time()
            for res in results:
                for orig, end, reff in res:
                    R_eff[orig, end] = reff
            myLogger.info(message=f"Collecting results took {time() - t_ss} seconds.")

        t_ss = time()
        with open(pkl_file_path, "wb") as f:
            pickle.dump(R_eff, f)
            myLogger.info(message=f"Reff.pkl saved. Took {time() - t_ss} seconds.")
        t_e = time()
        myLogger.info(message=f"Stage 3a took {t_e - t_s} seconds.")
    return R_eff


def stage1(dataset, isPygDataset=False, reuse=True):
    """
    Stage 1: Read the Dataset from pytorch geometric and write it into an .npz file
    Input:
        dataset: PygDataset or simple numpy array
        isPygDataset: True if dataset is a PygDataset
    """
    myLogger.info(message=f"Stage 1: converting pytorch dataset to npz file")
    if reuse and osp.exists(npz_file_path):
        myLogger.info(message=f"npz file already exists. Skipping...")
    else:
        myLogger.info(message=f"npz file not exist. Computing...")
        t_s = time()
        if isPygDataset:
            sparse_transform = ToSparseTensor()
            sparse_t_data = sparse_transform(dataset.data)
            scipy_data = sparse_t_data.adj_t.to_scipy(layout="csc")
            sparse.save_npz(npz_file_path, scipy_data)
            myLogger.info(message=f"npz file generated.")
        else:
            scipy_data = sparse.csc_matrix(
                (np.ones(dataset.shape[0], int), (dataset[:, 0], dataset[:, 1]))
            )
            sparse.save_npz(npz_file_path, scipy_data)
            myLogger.info(message=f"npz file generated.")
        t_e = time()
        myLogger.info(message=f"Stage 1 took {t_e - t_s} seconds.")


def stage2(reuse=True):
    """
    Stage 2: Run the Julia script that loads the npz file and generates the V matrix
    """
    myLogger.info(
        message=f"Stage 2: invoking julia script to generate V.csv matrix (Z in the paper)"
    )
    if reuse and osp.exists(csv_file_path):
        myLogger.info(message=f"csv file already exists. Skipping...")
    else:
        myLogger.info(message=f"csv file not exist. Computing...")
        t_s = time()
        cwd = os.getcwd()
        current_file_dir = osp.dirname(osp.realpath(__file__))
        os.chdir(current_file_dir)
        os.system(f"julia compute_V.jl --filepath={npz_file_path}")
        os.chdir(cwd)
        t_e = time()
        myLogger.info(message=f"Stage 2 took {t_e - t_s} seconds.")


def compute_edge_data(
    epsilon: Union[int, float],
    Pe,
    C,
    weights,
    start_nodes,
    end_nodes,
    N,
    dataset_name,
    config,
):
    """
    This function is called from stage3.
    This function is doing the final sampling based on Reff.
    This function is seperated to be able to use future.concurrent
    """
    if not isinstance(epsilon, int) and not isinstance(epsilon, float):
        myLogger.error(message=f"epsilon must be one of the type: int, float")
        sys.exit(1)
    myLogger.info(message=f"Stage 3b: sampling edges based on Reff")
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
    myLogger.info(message=f"Stage 3b took {t_e - t_s} seconds.")
    myLogger.info(
        f"Prune rate for epsilon={epsilon}: {1 - np.count_nonzero(new_weights) / np.size(new_weights)}, ({np.size(new_weights)} -> {np.count_nonzero(new_weights)})"
    )

    # convert into PyG's object
    edge_index, edge_weight = from_scipy_sparse_matrix(sparserW)
    if str(epsilon) in config[dataset_name]["python_er_epsilon_to_drop_rate_map"]:
        prune_file_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "edge_data.pt",
        )
        npy_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "dw.npy",
        )
        duw_el_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "duw.el",
        )
        dw_el_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "dw.wel",
        )
        uduw_el_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "uduw.el",
        )
        udw_el_path = osp.join(
            prune_file_dir,
            str(
                config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]
            ),
            "udw.wel",
        )

        t_s = time()
        os.makedirs(osp.dirname(npy_path), exist_ok=True)
        to_save = (
            torch.cat((edge_index, edge_weight.reshape(1, -1)), 0).numpy().transpose()
        )
        np.save(npy_path, to_save)
        myLogger.info(message=f"Saved edge_data.pt in {time() - t_s} seconds")

        # t_s = time()
        # os.makedirs(osp.dirname(npy_path), exist_ok=True)
        # to_save = torch.cat((edge_index, edge_weight.reshape(1, -1)), 0).numpy().transpose()
        # np.save(npy_path, to_save)
        # myLogger.info(message=f"Saved edge_data.pt in {time() - t_s} seconds")

    else:
        myLogger.info(
            message=f"Drop rate for epsilon={epsilon} not found in config, edge_data.pt not saved"
        )

    # if duw_el_path is not None and not osp.exists(duw_el_path):
    # os.makedirs(osp.dirname(duw_el_path), exist_ok=True)
    # with open(duw_el_path, 'w') as f:
    # for line in to_save:
    # f.write(f"{int(line[0])} {int(line[1])}\n")
    # myLogger.info(message=f"Saved edge list in {duw_el_path}")

    # if dw_el_path is not None and not osp.exists(dw_el_path):
    #     t = time()
    #     os.makedirs(osp.dirname(dw_el_path), exist_ok=True)
    #     with open(dw_el_path, 'w') as f:
    #         for line in to_save:
    #             f.write(f"{int(line[0])} {int(line[1])} {line[2]}\n")
    #     myLogger.info(message=f"Saved edge list in {dw_el_path} in {time() - t} seconds")

    # if uduw_el_path is not None and not osp.exists(uduw_el_path):
    #     os.makedirs(osp.dirname(uduw_el_path), exist_ok=True)
    #     with open(uduw_el_path, 'w') as f:
    #         for line in to_save:
    #             f.write(f"{int(line[0])} {int(line[1])}\n") if line[0] < line[1] else None
    #     myLogger.info(message=f"Saved edge list in {uduw_el_path}")

    # if udw_el_path is not None and not osp.exists(udw_el_path):
    #     os.makedirs(osp.dirname(udw_el_path), exist_ok=True)
    #     with open(udw_el_path, 'w') as f:
    #         for line in to_save:
    #             f.write(f"{int(line[0])} {int(line[1])} {line[2]}\n") if line[0] < line[1] else None
    #     myLogger.info(message=f"Saved edge list in {udw_el_path}")

    return edge_index, edge_weight


def stage3(
    dataset,
    dataset_name,
    epsilon: Union[int, float, list],
    config,
    isPygDataset=False,
    max_workers=64,
    reuse=True,
):
    myLogger.info(message=f"Stage 3: Pruning edges")
    if reuse and osp.exists(stage3_file_path):
        myLogger.info(message="stage3.npz already exists, loading")
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

        if isPygDataset:
            L_edge_idx, L_edge_attr = get_laplacian(dataset.data.edge_index)
        else:
            L_edge_idx, L_edge_attr = get_laplacian(torch.tensor(np.transpose(dataset)))
        L = SparseTensor(
            row=L_edge_idx[0], col=L_edge_idx[1], value=L_edge_attr
        )  # Lapacian, nxn
        L_scipy = L.to_scipy(layout="csc")
        L_diag = csc_matrix(
            (L_scipy.diagonal(), (np.arange(N), np.arange(N))), shape=(N, N)
        )
        W_scipy = L_diag - L_scipy  # Weight matrix, nxn

        # compute Reff
        R_eff = compute_reff(W_scipy, V, parallel=True, reuse=reuse)

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
        myLogger.info(message=f"stage3.npz saved")

    # Sampling
    # Rudelson, 1996 Random Vectors in the Isotropic Position
    # (too hard to figure out actual C0)
    C0 = 1 / 30.0
    # Rudelson and Vershynin, 2007, Thm. 3.1
    C = 4 * C0

    if isinstance(epsilon, int) or isinstance(epsilon, float):
        edge_index, edge_weight = compute_edge_data(
            epsilon, Pe, C, weights, start_nodes, end_nodes, N, dataset_name, config
        )
    elif isinstance(epsilon, list):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    compute_edge_data,
                    epsilon_,
                    Pe,
                    C,
                    weights,
                    start_nodes,
                    end_nodes,
                    N,
                    dataset_name,
                    config,
                ): epsilon_
                for epsilon_ in epsilon
            }
            for future in futures:
                print(f"Epsilon: {futures[future]}")
                future.result()
        edge_index, edge_weight = None, None
    else:
        myLogger.error(message=f"epsilon must be one of the type: int, float, list")
        sys.exit(1)
    return edge_index, edge_weight


def python_er_sparsify(
    dataset, dataset_name, epsilon: Union[int, float, list], config, reuse=True
):
    """
    This is the original ER sparsifier utilizing the Laplacian.jl repo
    Input:
        dataset: PygDataset
        dataset_name: str, name of the dataset
        epsilon: int | float -> return edge_index, edge_weight
                 list -> compute each epsilon, no return
        config: config dict
        reuse: bool, whether to reuse the previous computation
    Output:
        dataset: PygDataset with edge pruned
    """
    global npz_file_path, csv_file_path, pkl_file_path, stage3_file_path, prune_file_path, prune_file_dir
    npz_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/V.npz"
    )
    csv_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/V.csv"
    )
    pkl_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/Reff.pkl"
    )
    stage3_file_path = osp.join(
        osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/stage3.npz"
    )

    prune_file_dir = osp.join(
        osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/pruned/python_er"
    )
    os.makedirs(prune_file_dir, exist_ok=True)

    if isinstance(epsilon, int) or isinstance(epsilon, float):
        # if str(epsilon) in config[dataset_name]["python_er_epsilon_to_drop_rate_map"]:
        #     prune_file_path = osp.join(prune_file_dir, str(config[dataset_name]["python_er_epsilon_to_drop_rate_map"][str(epsilon)]), "edge_data.pt")
        myLogger.info(message=f"python_er_sparsify: epsilon: {epsilon}")
        if reuse and prune_file_path and osp.exists(prune_file_path):
            myLogger.info(message=f"edge_data.pt already exists. Loading it...")
            edge_data = torch.load(prune_file_path)
            edge_index = edge_data["edge_index"]
            edge_weight = edge_data["edge_weight"]
        else:
            myLogger.info(message=f"edge_data.pt does not exist. Computing it...")
            if dataset_name in ["Reddit", "Reddit2", "ogbn_products"]:
                stage1(
                    dataset.copy(), isPygDataset=True, reuse=reuse
                )  # stage1 will change dataset
                stage2(reuse=reuse)
                edge_index, edge_weight = stage3(
                    dataset,
                    dataset_name,
                    epsilon,
                    config,
                    isPygDataset=True,
                    reuse=reuse,
                )
            else:
                stage1(dataset.copy(), isPygDataset=False, reuse=reuse)
                stage2(reuse=reuse)
                edge_index, edge_weight = stage3(
                    dataset,
                    dataset_name,
                    epsilon,
                    config,
                    isPygDataset=False,
                    reuse=reuse,
                )
    elif isinstance(epsilon, list):
        myLogger.info(message=f"python_er_sparsify: epsilon: {epsilon}")
        if dataset_name in ["Reddit", "Reddit2", "ogbn_products"]:
            stage1(dataset.copy(), isPygDataset=True, reuse=reuse)
            stage2(reuse=reuse)
            edge_index, edge_weight = stage3(
                dataset, dataset_name, epsilon, config, isPygDataset=True, reuse=reuse
            )
        else:
            stage1(dataset.copy(), isPygDataset=False, reuse=reuse)
            stage2(reuse=reuse)
            edge_index, edge_weight = stage3(
                dataset, dataset_name, epsilon, config, isPygDataset=False, reuse=reuse
            )
    else:
        myLogger.error(message=f"epsilon must be one of the type: int, float, list")
        sys.exit(1)

    if dataset_name in ["Reddit", "Reddit2", "ogbn_products"]:
        data = dataset.data
        if edge_index is not None and edge_weight is not None:
            data.edge_index = edge_index
            data.edge_weight = edge_weight
        dataset.data = data
    return dataset


def cpp_er_sparsify(dataset_name, epsilon, config):
    """
    This is the gSparse version of ER pruning.

    Input:
        dataset_name: str, name of the dataset
        epsilon: int | float -> return edge_index, edge_weight
                 list -> compute each epsilon, no return
        config: config dict
    """
    if str(epsilon) in config[dataset_name]["cpp_er_epsilon_to_drop_rate_map"]:
        prune_file_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/pruned/er"
        )
        input_path = osp.join(
            osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/uduw.el"
        )
        er_path = osp.join(
            osp.dirname(osp.abspath(__file__)), f"../data/{dataset_name}/raw/er.txt"
        )
        el_path = osp.join(
            prune_file_dir,
            str(config[dataset_name]["cpp_er_epsilon_to_drop_rate_map"][str(epsilon)]),
            "udw.el_noweight",
        )
        weight_path = osp.join(
            prune_file_dir,
            str(config[dataset_name]["cpp_er_epsilon_to_drop_rate_map"][str(epsilon)]),
            "udw.weight",
        )
    os.makedirs(osp.dirname(el_path), exist_ok=True)
    Popen(
        osp.join(
            osp.dirname(osp.abspath(__file__)),
            f"./gSparse/bin/er {epsilon} {input_path} {el_path} {weight_path} {er_path}",
        ),
        shell=True,
    ).wait()
