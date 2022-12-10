import os
import os.path as osp
import sys
from time import time, time_ns
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix
from scipy import sparse, stats
from numpy import inf


npz_file_path = None
csv_file_path = None
pkl_file_path = None
stage3_file_path = None
prune_file_path = None
prune_file_dir = None

PRINT_REFF = False


def compute_reff(n, edges, V):
    R_eff = sparse.lil_matrix((n, n))
    for orig, end in edges:
        R_eff[orig, end] = np.linalg.norm(V[orig, :] - V[end, :]) ** 2

    if PRINT_REFF:
        print("\nReff:")
        for i in range(len(R_eff.rows)):
            for j in range(len(R_eff.rows[i])):
                print(f"{i}->{R_eff.rows[i][j]}: {R_eff.data[i][j]:.5f}")


def stage3(file_path):
    V_frame = pd.read_csv("Z.csv", header=None)
    V = V_frame.to_numpy()
    N = V.shape[0]

    # load edges
    t_s = time_ns()
    edges = []
    n, m, nnz = 0, 0, 0
    with open(file_path, "r") as f:
        lines = f.readlines()
        # n, m, nnz = [int(x) for x in lines[0].split()]
        # lines = lines[1:] # skip header
        for line in lines:
            line = line.strip().split()
            if line:
                src, dst = int(line[0]), int(line[1])
                n = max(n, src, dst)
                edges.append((src, dst))
    t_e = time_ns()
    print(f"read edge list time: {int((t_e - t_s)/1000000)} ms")

    n += 1
    # print(f"n: {n}, m: {m}, nnz: {nnz}")
    # print(f"V: {V}")
    # compute Reff
    t_s = time_ns()
    compute_reff(n, edges, V)
    t_e = time_ns()
    print(f"compute_reff time: {int((t_e - t_s)/1000000)} ms")


def main(argc, argv):
    # call julia
    total_t_s = time()

    t_s = time_ns()
    print("calling julia...")
    print(f"julia test.jl --filepath={argv[1]}")
    os.system(f"julia test.jl --filepath={argv[1]}")
    t_e = time_ns()
    print(f"julia script time: {int((t_e - t_s)/1000000)} ms")

    print("")
    stage3(argv[1])

    total_t_e = time()
    print(f"\ntotal time: {int(total_t_e - total_t_s)} s")


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
