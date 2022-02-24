import os
import sys
import os.path as osp
import json
import tempfile
import numpy as np
import torch
import myLogger

def compile_degree_pruner():
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    bin_path = os.path.join(current_file_dir, 'bin')
    os.makedirs(bin_path, exist_ok=True)

    os.chdir(current_file_dir)
    os.system(f'make -f {current_file_dir}/Makefile')
    os.chdir(cwd)


def degree_sparsify(dataset, dataset_name, in_or_out, degree_thres, config=None, recompile=False):
    """
    Input:
        dataset: PygDataset
        dataset_name: str, name of dataset 
        in_or_out: str, 'in' or 'out'
        degree_thres: int, threshold for in-degree
        config: loaded from config.json
    Output:
        dataset: PygDataset, with edges dropped
    """
    assert in_or_out in ['in', 'out'], 'in_or_out should be "in" or "out"'
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    myLogger.info(f'Getting {in_or_out}_degree prune file with threshold {degree_thres} for {dataset_name}')
    if config == None or str(degree_thres) not in config[dataset_name]['degree_thres_to_drop_rate_map']:
        myLogger.info(message=f'No config found for {dataset_name} with threshold {degree_thres}, edge_selection will be generated, but will not be saved to file')
        prune_file_path = None
    else:
        prune_file_path = os.path.join(current_file_dir, 
                                       f'../data/{dataset_name}/pruned/{in_or_out}_degree/', 
                                       str(config[dataset_name]['degree_thres_to_drop_rate_map'][str(degree_thres)]), 
                                       'edge_selection.npy')

    if prune_file_path is not None and osp.exists(prune_file_path):
        myLogger.info(message=f'Prune file already exists, loading edge selection')
        edge_selection = torch.load(prune_file_path)
    else:
        myLogger.info(message=f'Prune file not exist, generating {in_or_out}_degree prune file with threshold {degree_thres}')
        if recompile:
            compile_degree_pruner()
        edge_list_path = os.path.join(current_file_dir, f'../data/{dataset_name}/raw/edge_list.el')  
        tmpfile = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        os.chdir(current_file_dir)
        os.system(f'./bin/degree_prune -f {edge_list_path} -q {in_or_out}_threshold -x {degree_thres} -o {tmpfile.name}')
        os.chdir(cwd)
        if dataset_name in ['Reddit', 'Reddit2', 'ogbn_products']:
            edge_selection = torch.ones(dataset.data.edge_index.shape[1]).type(torch.bool)
            edge_selection[np.loadtxt(tmpfile.name)] = False
        else:
            myLogger.error(message=f'{dataset_name} is not supported. Exiting...')
            sys.exit(1)
        if prune_file_path is not None:
            os.makedirs(osp.dirname(prune_file_path), exist_ok=True)
            torch.save(edge_selection, prune_file_path)
            myLogger.info(message=f'Prune file saved for future use')

    data = dataset.data
    data.edge_index = data.edge_index[:, edge_selection]
    dataset.data = data
    return dataset


def in_degree_sparsify(dataset, dataset_name, degree_thres, config=None):
    return degree_sparsify(dataset, dataset_name, 'in', degree_thres, config=config)


def out_degree_sparsify(dataset, dataset_name, degree_thres, config=None):
    return degree_sparsify(dataset, dataset_name, 'out', degree_thres, config=config)