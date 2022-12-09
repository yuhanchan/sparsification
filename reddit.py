import sparsifier
import os
from os import path as osp
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import myLogger
import workload
import dataLoader

# setup logger
myLogger.set_level("INFO")
myLogger.get_logger(__name__)

reddit = dataLoader.Reddit()
# random
if False:
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {}
        for p in [
            0.32,  # 0.1
            0.45,  # 0.2
            0.55,  # 0.3
            0.63,  # 0.4
            0.71,  # 0.5
            0.775,  # 0.6
            0.837,  # 0.7
            0.894,  # 0.8
            0.95,  # 0.9
            0.975,  # 0.95
        ]:
            futures[
                executor.submit(
                    sparsifier.random_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/Reddit/raw/duw.el",
                    ),
                    dataset_name="Reddit",
                    prune_rate=p,
                    dataset_type="el",
                    post_symmetrize=True,
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
            18,  # 0.99
            63,  # 0.95
            115,  # 0.9
            226,  # 0.8
            354,  # 0.7
            509,  # 0.6
            701,  # 0.5
            953,  # 0.4
            1315,  # 0.3
            1900,  # 0.2
            3100,  # 0.1
        ]:
            futures[
                executor.submit(
                    sparsifier.sym_degree_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/Reddit/raw/duw.el",
                    ),
                    dataset_name="Reddit",
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
if True:
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        for epsilon in [5]:
            futures[
                executor.submit(
                    sparsifier.python_er_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/Reddit/raw/duw.el",
                    ),
                    dataset_name="Reddit",
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
