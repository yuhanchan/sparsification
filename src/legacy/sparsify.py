import sparsifier
import json
import os
from os import path as osp
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import myLogger
import workload

dataset_name = "ogbn_products" # Reddit, ogbn_products, ogbn_proteins
prune_algo = "ermin" # sym_random, sym_degree, er

# ------------------------------------

# setup logger
logger = myLogger.setup_custom_logger("root")
logger.debug("debug message")

config = json.load(open(f"{PROJECT_HOME}/config.json"))

# sym_random
if prune_algo == "sym_random":
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for key, val in config[dataset_name]["sym_random_to_prune_rate_map"].items():
            futures[
                executor.submit(
                    sparsifier.random_sparsify,
                    dataset_name=dataset_name,
                    prune_rate_key=key,
                    prune_rate_val=val,
                    dataset_type="el",
                    post_symmetrize=True,
                )
            ] = (key, val)

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)

# sym_degree
if prune_algo == "sym_degree":
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {}
        for key, val in config[dataset_name][
            "sym_degree_thres_to_prune_rate_map"
        ].items():
            futures[
                executor.submit(
                    sparsifier.sym_degree_sparsify,
                    dataset_name=dataset_name,
                    degree_thres=key,
                    prune_rate_val=val,
                )
            ] = (key, val)

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)


# ermin
if prune_algo == "ermin":
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for epsilon, val in config[dataset_name][
            "ermin_epsilon_to_prune_rate_map"
        ].items():
            futures[
                executor.submit(
                    sparsifier.python_er_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        f"data/{dataset_name}/raw/duw.el",
                    ),
                    dataset_name=dataset_name,
                    dataset_type="el",
                    epsilon=float(epsilon),
                    prune_rate_val=val,
                    reuse=True,
                    method="min",
                )
            ] = epsilon

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)

# ermax
if prune_algo == "ermax":
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for epsilon, val in config[dataset_name][
            "ermax_epsilon_to_prune_rate_map"
        ].items():
            futures[
                executor.submit(
                    sparsifier.python_er_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        f"data/{dataset_name}/raw/duw.el",
                    ),
                    dataset_name=dataset_name,
                    dataset_type="el",
                    epsilon=float(epsilon),
                    prune_rate_val=val,
                    reuse=True,
                    method="max",
                )
            ] = epsilon

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)
