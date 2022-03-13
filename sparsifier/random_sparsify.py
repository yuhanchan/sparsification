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

def random_sparsify(dataset, dataset_name, drop_rate):
    """
    Input:
        dataset: PygDataset
        dataset_name: str, name of dataset
        drop_rate: float, between 0 and 1 
    Output:
        dataset: PygDataset, with edges randomly dropped
    """
    myLogger.info(f'Getting random prune file with prune rate {drop_rate} for {dataset_name}')
    prune_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/edge_selection.npy')
    duw_el_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/duw.el')
    uduw_el_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/uduw.el')
    if osp.exists(prune_file_path):
        myLogger.info(message=f'Prune file already exists, loading edge selection')
        edge_selection = torch.load(prune_file_path)
    else:
        myLogger.info(message=f'Prune file not exist, generating random prune file with drop rate {drop_rate}')
        if dataset_name in ['Reddit', 'Reddit2', 'ogbn_products']:
            size = dataset.data.edge_index.shape[1]
        else:
            myLogger.info(message=f'{dataset_name} is not supported. Exiting...')
            sys.exit(1)
        edge_selection = torch.tensor(np.random.binomial(1, 1.0 - drop_rate, size=size)).type(torch.bool)
        os.makedirs(osp.dirname(prune_file_path), exist_ok=True)
        torch.save(edge_selection, prune_file_path)
        myLogger.info(message=f'Prune file saved for future use')
        
    data = dataset.data
    data.edge_index = data.edge_index[:, edge_selection]
    if not osp.exists(duw_el_path):
        np.savetxt(duw_el_path, data.edge_index.numpy().transpose().astype(int), fmt='%i')
    if not osp.exists(uduw_el_path):
        to_save = data.edge_index.numpy().transpose().astype(int)
        with open(uduw_el_path, 'w') as f:
            for line in to_save:
                f.write(f'{line[0]} {line[1]}\n') if line[0] < line[1] else None
    dataset.data = data
    return dataset


def cpp_random_sparsify(dataset_name, drop_rate, config=None, directed=True):
    """
    Input:
        dataset_name: str, name of dataset 
        drop_rate: float, between 0 and 1
        config: loaded from config.json
        directed: bool, if True, take edge list as is
                        if False, symmetrize edge list
    """
    cwd = os.getcwd()
    current_file_dir = os.path.dirname(os.path.realpath(__file__))

    myLogger.info(f'Generatin random prune file with drop_rate {drop_rate} for {dataset_name}')
    if config == None or drop_rate not in config[dataset_name]['drop_rate']:
        myLogger.error(f'{drop_rate} not in config[{dataset_name}][drop_rate]')
        sys.exit(1)

    duw_el_path = os.path.join(current_file_dir, 
                            f'../data/{dataset_name}/pruned/random/', 
                            f'{drop_rate}/duw.el')
    uduw_el_path = os.path.join(current_file_dir, 
                            f'../data/{dataset_name}/pruned/random/', 
                            f'{drop_rate}/uduw.el')
    os.makedirs(os.path.dirname(duw_el_path), exist_ok=True)
    
    os.chdir(current_file_dir)
    if directed:
        input_el_path = os.path.join(current_file_dir, f'../data/{dataset_name}/raw/duw.el')  
        os.system(f'./bin/prune -f {input_el_path} -q random -p {drop_rate} -o {duw_el_path}')
    else:
        input_el_path = os.path.join(current_file_dir, f'../data/{dataset_name}/raw/uduw.el')
        os.system(f'./bin/prune -f {input_el_path} -q random -p {drop_rate} -o {uduw_el_path}')
    os.chdir(cwd)

def cpp_random_sparsify_all(dataset_name, config, directed=True):
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for drop_rate in config[dataset_name]['drop_rate']:
            futures[executor.submit(el_random_sparsify, dataset_name, drop_rate, config, directed)] = drop_rate
        for future in futures:
            print(f'start {futures[future]}')
            try:
                future.result()
            except Exception as e:
                print(e)
                print(f'failed {futures[future]}')
                sys.exit(1)
                