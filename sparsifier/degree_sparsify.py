import os
import logging
import sys
import os.path as osp
import numpy as np
import torch
from collections import defaultdict
import random
import time

# setup logger
logger = logging.getLogger("root")
logger.debug("submodule message")

# set env
PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ") 
    print("please source env.sh at the top level of the project")
    exit(1)


def compile_degree_pruner():
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    bin_path = os.path.join(current_file_dir, "bin")
    os.makedirs(bin_path, exist_ok=True)

    os.chdir(current_file_dir)
    os.system(f"make -f {current_file_dir}/Makefile")
    os.chdir(cwd)


def pyg_degree_sparsify(
    dataset,
    dataset_name,
    in_or_out,
    degree_thres,
    prune_rate_val,
):
    """
    Input:
        dataset: PygDataset object only
        dataset_name: str, name of dataset
        in_or_out: str, 'in' or 'out'
        degree_thres: int, threshold for in-degree
        prune_rate_val: float, between 0 and 1, = true purne rate,
                        used for save file path, prefix with thres_ if set to None
        post_symmetrize: bool, if True, post symmetrize pruned edge list
    Output:
        dataset: PygDataset, with edges dropped
    """
    assert in_or_out in ["in", "out"], 'in_or_out should be "in" or "out"'
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    logger.info(f"---------- Degree sparsify Begin -----------\n")
    logger.info(f"{in_or_out}_degree_sparsify with threshold {degree_thres}")

    if prune_rate_val is None:
        logger.warning(
            f"prune_rate_val is None, " + "folder will prefixed with 'thres_'"
        )
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/{in_or_out}_degree/",
            f"thres_{degree_thres}/duw.el",
        )
    else:
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/{in_or_out}_degree/",
            f"{prune_rate_val}/duw.el",
        )

    input_duw_el_path = os.path.join(
        current_file_dir, f"../data/{dataset_name}/raw/duw.el"
    )

    # tmpfile = tempfile.NamedTemporaryFile(mode="w+", delete=True)
    os.chdir(current_file_dir)
    os.system(
        f"./bin/prune -f {input_duw_el_path} -q {in_or_out}_threshold -x {degree_thres} -o {duw_el_path}"
    )
    os.chdir(cwd)

    original_num_edges = dataset.data.edge_index.shape[1]
    dataset.data.edge_index = torch.tensor(np.loadtxt(duw_el_path).transpose())

    new_num_edges = dataset.data.edge_index.shape[1]

    logger.info(
        f"dataset: {dataset_name}, degree_thres: {degree_thres}, actual prune rate: {1.0 - new_num_edges / original_num_edges}\n"
        + f"---------- Degree sparsify End -----------\n"
    )

    return dataset


def el_degree_sparsify(
    dataset_name,
    in_or_out,
    degree_thres,
    prune_rate_val,
):
    """
    Input:
        dataset: PygDataset object only
        dataset_name: str, name of dataset
        in_or_out: str, 'in' or 'out'
        degree_thres: int, threshold for in-degree
        prune_rate_val: float, between 0 and 1, = true purne rate,
                        used for save file path, prefix with thres_ if set to None
        post_symmetrize: bool, if True, post symmetrize pruned edge list
    Output:
        dataset: PygDataset, with edges dropped
    """
    dataset = osp.join(PROJECT_HOME, f"data/{dataset_name}/raw/duw.el")
    assert in_or_out in ["in", "out"], 'in_or_out should be "in" or "out"'
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    logger.info(f"---------- Degree sparsify Begin -----------\n")
    logger.info(f"{in_or_out}_degree_sparsify with threshold {degree_thres}")

    if prune_rate_val is None:
        logger.warning(
            f"prune_rate_val is None, " + "folder will prefixed with 'thres_'"
        )
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/{in_or_out}_degree/",
            f"thres_{degree_thres}/duw.el",
        )
    else:
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/{in_or_out}_degree/",
            f"{prune_rate_val}/duw.el",
        )

    os.makedirs(os.path.dirname(duw_el_path), exist_ok=True)
    os.chdir(current_file_dir)
    os.system(
        f"./bin/prune -f {dataset} -q {in_or_out}_threshold -x {degree_thres} -o {duw_el_path}"
    )
    os.chdir(cwd)

    logger.info(f"---------- Degree sparsify End -----------\n")


def in_degree_sparsify(
    dataset_name,
    dataset_type,
    degree_thres,
    prune_rate_val,
):
    degree_thres = int(degree_thres)
    if dataset_type == "el":
        return el_degree_sparsify(
            dataset_name=dataset_name,
            in_or_out="in",
            degree_thres=degree_thres,
            prune_rate_val=prune_rate_val,
        )
    # elif dataset_type == "pyg":
    #     return pyg_degree_sparsify(
    #         dataset=dataset,
    #         dataset_name=dataset_name,
    #         in_or_out="in",
    #         degree_thres=degree_thres,
    #         prune_rate_val=prune_rate_val,
    #     )
    else:
        logger.error(f"dataset type {dataset_type} is not supported. Exiting...")
        sys.exit(1)


def out_degree_sparsify(
    dataset_name,
    dataset_type,
    degree_thres,
    prune_rate_val,
):
    degree_thres = int(degree_thres)
    if dataset_type == "el":
        return el_degree_sparsify(
            dataset_name=dataset_name,
            in_or_out="out",
            degree_thres=degree_thres,
            prune_rate_val=prune_rate_val,
        )
    # elif dataset_type == "pyg":
    #     return pyg_degree_sparsify(
    #         dataset=dataset,
    #         dataset_name=dataset_name,
    #         in_or_out="out",
    #         degree_thres=degree_thres,
    #         prune_rate_val=prune_rate_val,
    #     )
    else:
        logger.error(f"dataset type {dataset_type} is not supported. Exiting...")
        sys.exit(1)


def sym_degree_sparsify(
    dataset_name,
    degree_thres,
    prune_rate_val,
):
    """
    Input:
        dataset: PygDataset object only
        dataset_name: str, name of dataset
        in_or_out: str, 'in' or 'out'
        degree_thres: int, threshold for in-degree
        prune_rate_val: float, between 0 and 1, = true purne rate,
                        used for save file path, prefix with thres_ if set to None
        post_symmetrize: bool, if True, post symmetrize pruned edge list
    Output:
        dataset: PygDataset, with edges dropped
    """
    degree_thres = int(degree_thres)
    dataset = osp.join(PROJECT_HOME, f"data/{dataset_name}/raw/duw.el")
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    logger.info(f"---------- Degree sparsify Begin -----------\n")
    logger.info(f"sym_degree_sparsify with threshold {degree_thres}")

    if prune_rate_val is None:
        logger.warning(
            f"prune_rate_val is None, " + "folder will prefixed with 'thres_'"
        )
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/sym_degree/",
            f"thres_{degree_thres}/duw.el",
        )
    else:
        duw_el_path = os.path.join(
            current_file_dir,
            f"../data/{dataset_name}/pruned/sym_degree/",
            f"{prune_rate_val}/duw.el",
        )

    t_s = time.perf_counter()
    # read in duw.el
    edges = defaultdict(set)
    edge_list = np.loadtxt(dataset, dtype=int)
    for edge in edge_list:
        edges[edge[0]].add(edge[1])
    # remove self edge
    for src in edges:
        if src in edges[src]:
            edges[src].remove(src)
    print(
        f"Read graph done. # nodes: {len(edges)}, time: {time.perf_counter() - t_s} s"
    )
    original_num_edges = len(edge_list)

    # sym degree sparsify
    t_s = time.perf_counter()
    for src in edges:
        # if degree of src is larger than threshold, drop random edges
        while len(edges[src]) > degree_thres:
            # randomly drop an edge and sym drop
            dst = random.choice(list(edges[src]))
            edges[src].remove(dst)
            edges[dst].remove(src)
        # if not src % 10000:
        # print(f"{src}/{len(edges)} done")
    print(f"Degree sparsify done. time: {time.perf_counter() - t_s} s")

    # write out to duw_el_path
    t_s = time.perf_counter()
    os.makedirs(osp.dirname(duw_el_path), exist_ok=True)
    with open(duw_el_path, "w") as f:
        for src in edges:
            for dst in edges[src]:
                f.write(f"{src} {dst}\n")
    # sum up len of edge
    new_num_edges = 0
    for src in edges:
        new_num_edges += len(edges[src])
    print(f"Write out done. time: {time.perf_counter() - t_s} s")

    print(
        f"dataset: {dataset_name}, degree_thres: {degree_thres}, actual prune rate: {1.0 - new_num_edges / original_num_edges}\n"
        + f"---------- Degree sparsify End -----------\n"
    )
