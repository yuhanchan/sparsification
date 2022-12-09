import json
import os
import sys
import os.path as osp
import numpy as np

import shlex
import subprocess
import multiprocessing
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import myLogger
import sparsifier
import dataLoader
import workload

import argparse
from typing import Union


# def call_shell_cmd(i, cmd):
#     myLogger.info(f'Process {i} started')
#     myLogger.info(f'Process {i} cmd: {cmd}')
#     p = subprocess.Popen(cmd, shell=True)
#     our, err = p.communicate()
#     myLogger.info(f'Process {i} finished')
#     return (i, our, err)


def init_data():
    """
    This function generates all pruned dataset files with parameters specified in config.json.
    This function need only run once on new machine.
    """
    # setup logger
    myLogger.set_level("INFO")
    myLogger.get_logger(__name__)

    # Load config file
    myLogger.info(f"Loading config")
    config = json.load(open("config.json"))

    reddit = dataLoader.Reddit()
    reddit2 = dataLoader.Reddit2()
    ogbn_products = dataLoader.ogbn_products()

    # # random prune
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = {executor.submit(sparsifier.random_sparsify, dataset=reddit, dataset_name="Reddit", drop_rate=drop_rate): drop_rate for drop_rate in config["Reddit"]['drop_rate']}
    #     futures.update({executor.submit(sparsifier.random_sparsify, dataset=reddit2, dataset_name="Reddit2", drop_rate=drop_rate): drop_rate for drop_rate in config["Reddit2"]['drop_rate']})
    #     futures.update({executor.submit(sparsifier.random_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", drop_rate=drop_rate): drop_rate for drop_rate in config["ogbn_products"]['drop_rate']})

    #     for future in futures:
    #         drop_rate = futures[future]
    #         try:
    #             myLogger.info(f"Processing drop rate {drop_rate}")
    #             future.result()
    #         except:
    #             myLogger.error(f"Error processing drop rate {drop_rate}")
    #             sys.exit(1)
    #         else:
    #             myLogger.info(f"Finished with drop rate {drop_rate}")

    # # degree
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = {executor.submit(sparsifier.in_degree_sparsify, dataset=reddit, dataset_name="Reddit", degree_thres=thres, config=config): thres for thres in config["Reddit"]['degree_thres']}
    #     futures.update({executor.submit(sparsifier.in_degree_sparsify, dataset=reddit2, dataset_name="Reddit2", degree_thres=thres, config=config): thres for thres in config["Reddit2"]['degree_thres']})
    #     futures.update({executor.submit(sparsifier.in_degree_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", degree_thres=thres, config=config): thres for thres in config["ogbn_products"]['degree_thres']})

    #     futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=reddit, dataset_name="Reddit", degree_thres=thres, config=config): thres for thres in config["Reddit"]['degree_thres']})
    #     futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=reddit2, dataset_name="Reddit2", degree_thres=thres, config=config): thres for thres in config["Reddit2"]['degree_thres']})
    #     futures.update({executor.submit(sparsifier.out_degree_sparsify, dataset=ogbn_products, dataset_name="ogbn_products", degree_thres=thres, config=config): thres for thres in config["ogbn_products"]['degree_thres']})

    #     for future in futures:
    #         thres = futures[future]
    #         try:
    #             myLogger.info(f'Processing with thres {thres}')
    #             future.result()
    #         except:
    #             myLogger.error(f'Error with thres {thres}')
    #             sys.exit(1)
    #         else:
    #             myLogger.info(f'Finished with thres {thres}')

    # er
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                sparsifier.python_er_sparsify,
                dataset=reddit,
                dataset_name="Reddit",
                epsilon=epsilon,
                config=config,
            ): epsilon
            for epsilon in config["Reddit"]["python_er_epsilon"]
        }
        futures.update(
            {
                executor.submit(
                    sparsifier.python_er_sparsify,
                    dataset=reddit2,
                    dataset_name="Reddit2",
                    epsilon=epsilon,
                    config=config,
                ): epsilon
                for epsilon in config["Reddit2"]["python_er_epsilon"]
            }
        )
        futures.update(
            {
                executor.submit(
                    sparsifier.python_er_sparsify,
                    dataset=ogbn_products,
                    dataset_name="ogbn_products",
                    epsilon=epsilon,
                    config=config,
                ): epsilon
                for epsilon in config["ogbn_products"]["python_er_epsilon"]
            }
        )

        for future in futures:
            epsilon = futures[future]
            try:
                myLogger.info(f"Processing with epsilon {epsilon}")
                future.result()
            except:
                myLogger.error(f"Error with epsilon {epsilon}")
                sys.exit(1)
            else:
                myLogger.info(f"Finished with epsilon {epsilon}")


def main():
    # setup logger
    myLogger.set_level("INFO")
    myLogger.get_logger(__name__)

    # setup dir
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir_root = osp.join(current_file_dir, "experiments")
    os.makedirs(experiment_dir_root, exist_ok=True)

    # parse args
    parser = argparse.ArgumentParser(description="Top level script")
    parser.add_argument(
        "-w", "--workload", type=str, required=True, help="workload to run"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )
    parser.add_argument(
        "-s", "--sparsifier", type=str, required=True, help="sparsifier to use"
    )
    parser.add_argument(
        "-p",
        "--prune",
        type=str,
        required=True,
        help="Prune argument to use."
        + "For random, it's used as prune rate."
        + "For in_degree/out_degree, it's used as degree threshold."
        + "For er, it's used as epsilon."
        + "\nAlternatively, you can use 'all' to run all prune level in config.json.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Set this option to run prune arguments not in config.json",
    )
    args = parser.parse_args()

    # Sanity check
    assert args.workload in [
        "bc",
        "bfs",
        "cc",
        "cc_sv",
        "pr",
        "pr_spmv",
        "sssp",
        "tc",
        "ClusterGCN",
    ], f"Unknown workload: {args.workload}"
    assert args.dataset in [
        "Reddit",
        "Reddit2",
        "ogbn_products",
        "sk_2005",
    ], f"Unknown dataset: {args.dataset}"
    assert args.sparsifier in [
        "baseline",
        "random",
        "in_degree",
        "out_degree",
        "old_er",
        "er",
    ], f"Unknown sparsifier: {args.sparsifier}"

    # Load config file
    myLogger.info(f"Loading config")
    config = json.load(open("config.json"))

    # set prune levels
    if args.prune == "all":
        if args.sparsifier == "baseline":
            prune_levels = [args.prune]
        elif args.sparsifier == "random":
            prune_levels = config[args.dataset]["drop_rate"]
        elif args.sparsifier == "in_degree" or args.sparsifier == "out_degree":
            prune_levels = config[args.dataset]["degree_thres"]
        elif args.sparsifier == "er":
            prune_levels = config[args.dataset]["er_epsilon"]
        elif args.sparsifier == "old_er":
            prune_levels = config[args.dataset]["old_er_epsilon"]
        else:
            prune_levels = []
    else:
        prune_levels = [float(args.prune)] if "." in args.prune else [int(args.prune)]

    for prune_level in prune_levels:
        if args.workload == "ClusterGCN":
            # Load dataset
            if args.dataset == "Reddit":
                dataset = dataLoader.Reddit()
            elif args.dataset == "Reddit2":
                dataset = dataLoader.Reddit2()
            elif args.dataset == "ogbn_products":
                dataset = dataLoader.ogbn_products()
            else:
                myLogger.error(f"Unknown dataset: {args.dataset}")

            # Apply sparsifier
            if args.sparsifier == "baseline":
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}",
                )
            elif args.sparsifier == "random":
                if (
                    not args.force
                    and prune_level not in config[args.dataset]["drop_rate"]
                ):
                    myLogger.error(
                        f"Prune rate {prune_level} for random prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting..."
                    )
                    sys.exit(1)
                dataset = sparsifier.random_sparsify(
                    dataset=dataset, dataset_name=args.dataset, drop_rate=prune_level
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{prune_level}",
                )
            elif args.sparsifier == "in_degree":
                if (
                    not args.force
                    and prune_level not in config[args.dataset]["degree_threshold"]
                ):
                    myLogger.error(
                        f"Degree threshold {prune_level} for in_degree prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting..."
                    )
                    sys.exit(1)
                dataset = sparsifier.in_degree_sparsify(
                    dataset=dataset,
                    dataset_name=args.dataset,
                    degree_thres=prune_level,
                    config=config,
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['degree_thres_to_drop_rate_map'][str(prune_level)]}",
                )
            elif args.sparsifier == "out_degree":
                if (
                    not args.force
                    and prune_level not in config[args.dataset]["degree_threshold"]
                ):
                    myLogger.error(
                        f"Degree threshold {prune_level} for out_degree prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting..."
                    )
                    sys.exit(1)
                dataset = sparsifier.out_degree_sparsify(
                    dataset=dataset,
                    dataset_name=args.dataset,
                    degree_thres=prune_level,
                    config=config,
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['degree_thres_to_drop_rate_map'][str(prune_level)]}",
                )
            elif args.sparsifier == "er":
                if (
                    not args.force
                    and prune_level not in config[args.dataset]["er_epsilon"]
                ):
                    myLogger.error(
                        f"Epsilon {prune_level} for er prune for {args.dataset} not found in config.json, if you want to force this, please set -f. Exiting..."
                    )
                dataset = sparsifier.er_sparsify(
                    dataset=dataset,
                    dataset_name=args.dataset,
                    epsilon=prune_level,
                    config=config,
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{config[args.dataset]['er_epsilon_to_drop_rate_map'][str(prune_level)]}",
                )

            print(f"Experiment dir: {experiment_dir}")
            os.makedirs(experiment_dir, exist_ok=True)
            # Invoke workload

        else:
            # set dataset path
            if args.sparsifier == "baseline":
                duw_el_path = osp.join(
                    current_file_dir, "data", args.dataset, "raw", "duw.el"
                )
                uduw_el_path = osp.join(
                    current_file_dir, "data", args.dataset, "raw", "uduw.el"
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}",
                )
            elif args.sparsifier == "random":
                if prune_level not in config[args.dataset]["drop_rate"]:
                    myLogger.error(
                        f"Prune rate {prune_level} for random prune for {args.workload} not found in config.json. Exiting..."
                    )
                    sys.exit(1)
                duw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(prune_level),
                    "duw.el",
                )
                uduw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(prune_level),
                    "uduw.el",
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{prune_level}",
                )
            elif args.sparsifier == "in_degree" or args.sparsifier == "out_degree":
                if prune_level not in config[args.dataset]["degree_thres"]:
                    myLogger.error(
                        f"Degree threshold {prune_level} for in_degree/out_degree prune for {args.workload} not found in config.json. Exiting..."
                    )
                    sys.exit(1)
                drop_rate = config[args.dataset]["degree_thres_to_drop_rate_map"][
                    str(prune_level)
                ]
                duw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "duw.el",
                )
                uduw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "uduw.el",
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{drop_rate}",
                )
            elif args.sparsifier == "er":
                if prune_level not in config[args.dataset]["er_epsilon"]:
                    myLogger.error(
                        f"Epsilon {prune_level} for er prune for {args.workload} not found in config.json. Exiting..."
                    )
                    sys.exit(1)
                drop_rate = config[args.dataset]["er_epsilon_to_drop_rate_map"][
                    str(prune_level)
                ]
                dw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "dw.wel",
                )
                udw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "udw.wel",
                )
                duw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "duw.el",
                )
                uduw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "uduw.el",
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{drop_rate}",
                )
            elif args.sparsifier == "old_er":
                if prune_level not in config[args.dataset]["old_er_epsilon"]:
                    myLogger.error(
                        f"Epsilon {prune_level} for er prune for {args.workload} not found in config.json. Exiting..."
                    )
                    sys.exit(1)
                drop_rate = config[args.dataset]["old_er_epsilon_to_drop_rate_map"][
                    str(prune_level)
                ]
                dw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "dw.wel",
                )
                udw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "udw.wel",
                )
                duw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "duw.el",
                )
                uduw_el_path = osp.join(
                    current_file_dir,
                    "data",
                    args.dataset,
                    "pruned",
                    args.sparsifier,
                    str(drop_rate),
                    "uduw.el",
                )
                experiment_dir = osp.join(
                    experiment_dir_root,
                    f"{args.workload}/{args.dataset}/{args.sparsifier}/{drop_rate}",
                )

            os.makedirs(experiment_dir, exist_ok=True)

            # Invoke workload
            if args.workload == "bc":
                input_file_path = duw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.bc(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "bfs":
                input_file_path = duw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.bfs(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "cc":
                input_file_path = uduw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.cc(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "cc_sv":
                input_file_path = uduw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.cc_sv(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "pr":
                input_file_path = dw_el_path if args.sparsifier == "er" else duw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.pr(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "pr_spmv":
                input_file_path = dw_el_path if args.sparsifier == "er" else duw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.pr_spmv(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "sssp":
                input_file_path = (
                    dw_el_path
                    if args.sparsifier == "er" or args.sparsifier == "old_er"
                    else duw_el_path
                )
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.sssp(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-v": "",
                        "-a": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )
            elif args.workload == "tc":
                input_file_path = uduw_el_path
                assert osp.exists(
                    input_file_path
                ), f"Input file {input_file_path} does not exist. Exiting..."
                workload.tc(
                    **{
                        "-f": input_file_path,
                        "-n": "10",
                        "-a": "",
                        "-s": "",
                        "-z": osp.join(experiment_dir, "analysis.txt"),
                        ">": osp.join(experiment_dir, "stdout.txt"),
                    }
                )


if __name__ == "__main__":
    # main()

    myLogger.set_level("INFO")
    myLogger.get_logger(__name__)
    config = json.load(open("config.json"))
    current_file_dir = osp.dirname(osp.realpath(__file__))

    reddit = dataLoader.Reddit()
    sparsifier.python_er_sparsify(reddit, "Reddit", 2, config, reuse=False)

    # reddit2 = dataLoader.Reddit2()
    # sparsifier.python_er_sparsify(reddit2, "Reddit2", 2, config, reuse=False)

    # ogbn = dataLoader.ogbn_products()
    # sparsifier.python_er_sparsify(ogbn, "ogbn_products", 2.1, config, reuse=False)

    # dataset = np.loadtxt(osp.join(current_file_dir, 'data/sk-2005/sk-2005.el'), dtype=int)
    # workload.pr(**{"-f": f'{current_file_dir}/data/sk-2005/sk-2005.el', "-n": "1", "-v": ""})
    # workload.pr(**{"-f": f'{current_file_dir}/data/AGATHA_2015/AGATHA_2015.el', "-n": "1", "-v": ""})
    # sparsifier.el_random_sparsify('sk_2005', 0.9, config=config)
    # sparsifier.el_random_sparsify_all('sk_2005', config=config)
    # sparsifier.el_random_sparsify('AGATHA_2015', 0.1, config=config, directed=False)
    # sparsifier.el_random_sparsify_all('AGATHA_2015', config=config, directed=False)

    # reddit2 = dataLoader.Reddit2()
    # sparsifier.er_sparsify(reddit2, "Reddit2", 0.5, config)

    # with ProcessPoolExecutor(max_workers=32) as executor:
    #     futures = {}
    #     for epsilon in config['orkut']['cpp_er_epsilon']:
    #         futures[executor.submit(sparsifier.cpp_er_sparsify, "orkut", epsilon, config)] = epsilon
    #     for future in futures:
    #         try:
    #             print("Runing with epsilon: ", futures[future])
    #             future.result()
    #         except Exception as e:
    #             print(e)
    #             print("Error with epsilon: ", futures[future])
    #             sys.exit(1)

    # reddit = dataLoader.Reddit()
    # sparsifier.python_er_sparsify(reddit, "Reddit", epsilon=2, config=config)

    # ogbn = dataLoader.ogbn_products()
    # sparsifier.python_er_sparsify(ogbn, "ogbn_products", epsilon=2.1, config=config)

    # gd = np.load(osp.join(current_file_dir, 'data/gd/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(gd, 'gd', 0.5, config)

    # orkut = np.load(osp.join(current_file_dir, 'data/orkut/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(orkut, 'orkut', 0.5, config)

    # road_usa = np.load(osp.join(current_file_dir, 'data/road_usa/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(road_usa, 'road_usa', 0.5, config)

    # sk_2005 = np.load(osp.join(current_file_dir, 'data/sk_2005/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(sk_2005, 'sk_2005', 0.5, config) # took 27 hours and 1.3 TB mem at max

    # uk_2005 = np.load(osp.join(current_file_dir, 'data/uk_2005/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(uk_2005, 'uk_2005', 10, config)

    # friendster = np.load(osp.join(current_file_dir, 'data/friendster/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(friendster, 'friendster', 0.5, config)

    # AGATHA_2015 = np.load(osp.join(current_file_dir, 'data/AGATHA_2015/raw/duw.npy')).astype(int)
    # sparsifier.python_er_sparsify(AGATHA_2015, 'AGATHA_2015', 0.5, config)

    # test_dataset = np.loadtxt(osp.join(current_file_dir, '../linear_solver/A_10_18.txt')).astype(int)
    # sparsifier.python_er_sparsify(test_dataset, 'test_dataset', 0.5, config)

    # init_data()
