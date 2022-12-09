import sparsifier
import os
from os import path as osp
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import myLogger
import workload

# setup logger
myLogger.set_level("INFO")
myLogger.get_logger(__name__)

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
                        "data/web-Google/raw/web-Google.el",
                    ),
                    dataset_name="web-Google",
                    prune_rate=p,
                    dataset_type="el",
                    use_cache=True,
                    symmetrize=True,
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
            1,  # 0.97
            2,  # 0.93
            3,  # 0.88
            5,  # 0.8
            8,  # 0.7
            11,  # 0.6
            16,  # 0.5
            23,  # 0.4
            38,  # 0.3
            74,  # 0.2
            261,  # 0.1
        ]:
            futures[
                executor.submit(
                    sparsifier.sym_degree_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/web-Google/raw/web-Google.el",
                    ),
                    dataset_name="web-Google",
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
            0.317,  # 0.1
            0.406,  # 0.2
            0.493,  # 0.3
            0.587,  # 0.4
            0.7,  # 0.5
            0.844,  # 0.6
            1.04,  # 0.7
            1.36,  # 0.8
            2.04,  # 0.9
            2.97,  # 0.95
            4.8,  # 0.98
        ]:
            futures[
                executor.submit(
                    sparsifier.python_er_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/web-Google/raw/web-Google.el",
                    ),
                    dataset_name="web-Google",
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
                "web-Google",
                "pruned",
                prune_method,
            )
            for prune_rate in os.listdir(input_dir):
                input_file_path = osp.join(input_dir, prune_rate, "dw.wel")
                experiment_dir = osp.join(
                    osp.dirname(osp.realpath(__file__)),
                    "experiments",
                    "pr",
                    "web-Google",
                    f"{prune_method}_with_weight",
                    prune_rate,
                )
                print(f"{prune_method} web-Google {prune_rate}")
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
