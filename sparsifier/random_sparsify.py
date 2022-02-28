import numpy as np
import torch
import myLogger
import os.path as osp
import os
import sys

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
    edge_list_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/edge_list.el')
    undirected_edge_list_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/undirected_edge_list.el')
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
    if not osp.exists(edge_list_file_path):
        np.savetxt(edge_list_file_path, data.edge_index.numpy().transpose().astype(int), fmt='%i')
    if not osp.exists(undirected_edge_list_file_path):
        to_save = data.edge_index.numpy().transpose().astype(int)
        with open(undirected_edge_list_file_path, 'w') as f:
            for line in to_save:
                f.write(f'{line[0]} {line[1]}\n') if line[0] < line[1] else None
    dataset.data = data
    return dataset