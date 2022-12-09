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
# print(f"logger name : {__name__}")
# myLogger.info("hi")

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
                        "data/ogbn_products/raw/duw.el",
                    ),
                    dataset_name="ogbn_products",
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
            # 5,  # 0.96
            # 6,  # 0.95
            # 7,  # 0.94
            # 8,  # 0.93
            # 9,  # 0.92
            # 10, # 0.91
            11, # 0.9
            # 16, # 0.854
            # 23, # 0.7935
            # 38, # 0.6835
            # 74, # 0.5
            # 261, # 0.1756
            # 300, # 0.15
            # 340, # 0.13
            # 390, # 0.11
            420, # 0.1
        ]:
            futures[
                executor.submit(
                    sparsifier.sym_degree_sparsify,
                    osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        "data/ogbn_products/raw/duw.el",
                    ),
                    dataset_name="ogbn_products",
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
        for epsilon in [
            0.132,  # 0.1
            # 0.885,  # 0.9
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

