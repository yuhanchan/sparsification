import numpy as np
import torch
import myLogger
import os.path as osp
import os
import sys
from concurrent.futures import ProcessPoolExecutor


# set random seed for deterministic results
torch.manual_seed(1)
np.random.seed(1)


def pyg_random_sparsify(dataset, dataset_name, prune_rate, use_cache=True, save_edgelist=True, symmetrize=True):
    """
    Input:
        dataset: PygDataset object only
        dataset_name: str, name of dataset
        prune_rate: float, between 0 and 1
        use_cache: bool, if True, use cached edge selection file. default is True
        save_edgelist: bool, if True, save raw edgelist file, raw edgelist is used some workloads. default is True
        symmetrize: bool, if True, symmetrize pruned edge list
    Output:
        dataset: PygDataset, with edges randomly pruned
    """
    myLogger.info(
        f"---------- Random sparsify Begin -----------\n"
        + f"dataset: {dataset_name}, target prune rate: {prune_rate}"
    )
    if symmetrize:
        myLogger.info("pruned edge list will be symmetrized, make sure the input edgelist is symmetrical, otherwise the result prune rate will be incorrect!")

    edge_selection_file = osp.join(
        osp.dirname(osp.abspath(__file__)),
        f"../data/{dataset_name}/pruned/random/{prune_rate}/edge_selection.npy",
    )
    duw_el_path = osp.join(
        osp.dirname(osp.abspath(__file__)),
        f"../data/{dataset_name}/pruned/random/{prune_rate}/duw.el",
    )
    # uduw_el_path = osp.join(
        # osp.dirname(osp.abspath(__file__)),
        # f"../data/{dataset_name}/pruned/random/{prune_rate}/uduw.el",
    # )

    if osp.exists(edge_selection_file) and use_cache:
        myLogger.info(message=f"Edge selection file already exists, loading edge selection")
        edge_selection = torch.load(edge_selection_file)
    else:
        myLogger.info(
            message=f"Edge selection file not exist, generating random prune file with drop rate {prune_rate}"
        )
        if dataset_name in ["Reddit", "Reddit2", "ogbn_products"]:
            size = dataset.data.edge_index.shape[1]
        else:
            myLogger.info(message=f"{dataset_name} is not supported. Exiting...")
            sys.exit(1)
        edge_selection = torch.tensor(
            np.random.binomial(1, 1.0 - prune_rate, size=size)
        ).type(torch.bool)
        os.makedirs(osp.dirname(edge_selection_file), exist_ok=True)
        torch.save(edge_selection, edge_selection_file)
        myLogger.info(message=f"Prune file saved for future use")

    data = dataset.data
    original_num_edges = data.edge_index.shape[1]

    if symmetrize: # pick src < dst edges
        data.edge_index = data.edge_index[:, edge_selection]
        edge_index = data.edge_index[:, data.edge_index[1, :] < data.edge_index[0, :]]
        # symmetrize
        data.edge_index = torch.vstack((torch.cat((edge_index[0, :], edge_index[1, :]), 0), torch.cat((edge_index[1, :], edge_index[0, :]), 0)))
    else:
        data.edge_index = data.edge_index[:, edge_selection]
    new_num_edges = data.edge_index.shape[1]

    if save_edgelist:
        os.makedirs(osp.dirname(duw_el_path), exist_ok=True)
        # os.makedirs(osp.dirname(uduw_el_path), exist_ok=True)
        to_save = data.edge_index.numpy().transpose().astype(int)
        np.savetxt(duw_el_path, to_save, fmt="%i")
        # with open(uduw_el_path, "w") as f:
            # for line in to_save:
                # f.write(f"{line[0]} {line[1]}\n") if line[0] < line[1] else None

    dataset.data = data

    myLogger.info(
        f"dataset: {dataset_name}, target prune rate: {prune_rate}, actual prune rate: {1.0 - new_num_edges / original_num_edges}\n"
        + f"---------- Random sparsify End -----------\n"
    )

    return dataset


def el_random_sparsify(dataset, dataset_name, prune_rate, use_cache=True, save_edgelist=True, symmetrize=True):
    """
    Input:
        dataset: str, path to edgelist file, must be absolute path
        dataset_name: str, name of dataset
        prune_rate: float, between 0 and 1
        use_cache: not used in this functin, kept for compatibility
        save_edgelist: not used in this functin, kept for compatibility
        symmetrize: bool, if True, symmetrize pruned edge list
    Output:
        None, pruned edgelist file saved
    """
    # sanity checks
    if not isinstance(dataset, str):
        raise ValueError(f"dataset must be str, got {type(dataset)}")
    if not osp.isabs(dataset):
        raise ValueError(f"dataset must be absolute path, got {dataset}")
    if not osp.exists(dataset):
        raise ValueError(f"dataset {dataset} does not exist")

    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    myLogger.info(
        f"---------- Random sparsify -----------\n"
        + f"dataset: {dataset_name}, prune_rate: {prune_rate}"
    )
    if symmetrize:
        myLogger.info("pruned edge list will be symmetrized, make sure the input edgelist is symmetrical, otherwise the result prune rate will be incorrect!")

    # if config == None or prune_rate not in config[dataset_name]["prune_rate"]:
        # myLogger.error(f"{prune_rate} not in config[{dataset_name}][prune_rate]")
        # sys.exit(1)

    duw_el_path = os.path.join(
        current_file_dir,
        f"../data/{dataset_name}/pruned/random/",
        f"{prune_rate}/duw.el",
    )
    # uduw_el_path = os.path.join(
        # current_file_dir,
        # f"../data/{dataset_name}/pruned/random/",
        # f"{prune_rate}/uduw.el",
    # )
    os.makedirs(os.path.dirname(duw_el_path), exist_ok=True)
    # os.makedirs(os.path.dirname(uduw_el_path), exist_ok=True)

    os.chdir(current_file_dir)
    if symmetrize:
        os.system(
            f"./bin/prune -f {dataset} -q random -p {prune_rate} -o {duw_el_path} -s"
        )
    else:
        os.system(
            f"./bin/prune -f {dataset} -q random -p {prune_rate} -o {duw_el_path}"
        )
    os.chdir(cwd)


def cpp_random_sparsify_all(dataset_name, config, directed=True):
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for prune_rate in config[dataset_name]["prune_rate"]:
            futures[
                executor.submit(
                    el_random_sparsify, dataset_name, prune_rate, config, directed
                )
            ] = prune_rate
        for future in futures:
            print(f"start {futures[future]}")
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f"failed {futures[future]}")
                sys.exit(1)


# This is the top level function
def random(dataset, dataset_name, prune_rate, dataset_type="el", use_cache=True, save_edgelist=True, symmetrize=True):
    """
    Input:
        dataset: str, path to edgelist file, must be absolute path
        dataset_name: str, name of dataset
        prune_rate: float, between 0 and 1
        dataset_type: str, [el/pyg], el for edgelist, pyg for pyg object
        use_cache: not used in this functin, kept for compatibility
        save_edgelist: not used in this functin, kept for compatibility
        symmetrize: bool, if True, symmetrize pruned edge list
    """
    if dataset_type == "el":
        myLogger.info("---------- Random sparsify for el format input -----------")
        el_random_sparsify(dataset, dataset_name, prune_rate, use_cache=use_cache, save_edgelist=save_edgelist, symmetrize=symmetrize)
    elif dataset_type == "pyg":
        myLogger.info("---------- Random sparsify for pyg format input -----------")
        pyg_random_sparsify(dataset, dataset_name, prune_rate, use_cache=use_cache, save_edgelist=save_edgelist, symmetrize=symmetrize)
    else:
        raise ValueError(f"dataset_type must be el/pyg, got {dataset_type}")
