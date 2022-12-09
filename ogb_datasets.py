from os import wait
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import sparsifier
import os
from os import path as osp
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import myLogger
import workload
import dataLoader

dataset_name = "ogbn-products"

# dataset = PygNodePropPredDataset(
#     name=dataset_name,
#     transform=T.ToSparseTensor(),
#     root="data/",
# )


# setup logger
myLogger.set_level("INFO")
myLogger.get_logger(__name__)

# random
if True:
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {}
        for p in [
            # 0.1,
            # 0.2,
            # 0.3,
            # 0.4,
            # 0.5,
            # 0.6,
            # 0.7,
            # 0.8,
            # 0.9,
            # 0.95,
            0.96,
            0.97,
            0.98,
        ]:
            futures[
                executor.submit(
                    sparsifier.random_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/ogbn_products/raw/uduw.el",
                    ),
                    dataset_name="ogbn_products",
                    prune_rate=p,
                    dataset_type="el",
                    post_symmetrize=False,
                )
            ] = p

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)

# degree
if False:
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for d in [
            1,  # 0.966
            2,  # 0.935
            3,  # 0.9
            8,  # 0.8
            14,  # 0.7
            23,  # 0.6
            36,  # 0.5
            56,  # 0.4
            89,  # 0.3
            148,  # 0.2
            291,  # 0.1
        ]:
            futures[
                executor.submit(
                    sparsifier.out_degree_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/ogbn_products/raw/uduw.el",
                    ),
                    dataset_name="ogbn_products",
                    dataset_type="el",
                    degree_thres=d,
                    config=None,
                )
            ] = d

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)

# er
if False:
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for epsilon in [
            0.132,  # 0.1
            0.17,  # 0.2
            0.206,  # 0.3
            0.246,  # 0.4
            0.294,  # 0.5
            0.356,  # 0.6
            0.442,  # 0.7
            0.581,  # 0.8
            0.885,  # 0.9
            1.3,  # 0.95
            1.47,  # 0.96
            1.71,  # 0.97
            2.1,  # 0.98
        ]:
            futures[
                executor.submit(
                    sparsifier.python_er_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/ogbn_products/raw/duw.el",
                    ),
                    dataset_name="ogbn_products",
                    dataset_type="el",
                    epsilon=epsilon,
                    config=None,
                    reuse=True,
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


# symmetrize random and degree pruned graphs
def call_utils(input_path, output_path):
    os.system(f"./utils/bin/utils -i {input_path} -o {output_path} -m 11")


if False:
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}

        for prune_method in ["random", "sym_degree"]:
            for prune_rate in os.listdir(f"data/ogbn_products/pruned/{prune_method}"):
                folder = f"data/ogbn_products/pruned/{prune_method}/{prune_rate}"
                os.system(f"mv {folder}/duw.el {folder}/uduw.el")
                input_file = f"{folder}/uduw.el"
                output_file = f"{folder}/duw.el"
                print(f"symmetrizing {input_file}")
                futures[executor.submit(call_utils, input_file, output_file)] = (
                    prune_method,
                    prune_rate,
                )

        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)


# run PR
if False:
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        # for prune_method in ["random", "sym_degree", "er"]:
        for prune_method in ["er"]:
            input_dir = osp.join(
                osp.dirname(osp.realpath(__file__)),
                "data",
                "Reddit",
                "pruned",
                prune_method,
            )
            for prune_rate in os.listdir(input_dir):
                input_file_path = osp.join(input_dir, prune_rate, "dw.wel")
                experiment_dir = osp.join(
                    osp.dirname(osp.realpath(__file__)),
                    "experiments",
                    "pr",
                    "Reddit",
                    f"{prune_method}_with_weight",
                    prune_rate,
                )
                print(f"{prune_method} Reddit {prune_rate}")
                os.makedirs(experiment_dir, exist_ok=True)
                futures[
                    executor.submit(
                        workload.pr,
                        **{
                            "-f": input_file_path,
                            "-n": "1",
                            "-v": "",
                            "-a": "",
                            "-z": osp.join(experiment_dir, "analysis.txt"),
                            ">": osp.join(experiment_dir, "stdout.txt"),
                        },
                    )
                ] = (prune_method, prune_rate)
        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)
