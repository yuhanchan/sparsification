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
from scipy import sparse, stats
from numpy import inf


npz_file_path = None
csv_file_path = None
pkl_file_path = None
stage3_file_path = None
prune_file_path = None
prune_file_dir = None

def compute_reff(n, edges, V):
    print(V)
    R_eff = sparse.lil_matrix((n, n))
    for orig, end in edges:
        # print(f"{orig} -> {end}, {V[orig, :] - V[end, :]}")
        R_eff[orig, end] = np.linalg.norm(V[orig, :] - V[end, :]) ** 2

    print("\nReff:")
    for i in range(len(R_eff.rows)):
        for j in range(len(R_eff.rows[i])):
            print(f"{i}->{R_eff.rows[i][j]}: {R_eff.data[i][j]:.5f}")


def stage3(file_path):
    V_frame = pd.read_csv("Z.csv", header=None)
    V = V_frame.to_numpy()
    N = V.shape[0]

    # load edges
    edges = []
    n, m, nnz = 0, 0, 0
    with open(file_path, "r") as f:
        lines = f.readlines()
        n, m, nnz = [int(x) for x in lines[0].split()]
        lines = lines[1:] # skip header
        for line in lines:
            line = line.strip().split()
            if line:
                edges.append((int(line[0]), int(line[1])))

    # compute Reff 
    R_eff = compute_reff(n, edges, V)

def main(argc, argv):
    # call julia
    os.system(f"julia test.jl --filepath={argv[1]}")
    
    print("")
    stage3(argv[1])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)

