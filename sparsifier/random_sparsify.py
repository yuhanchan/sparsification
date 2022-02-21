import numpy as np
import torch
import myLogger
import os.path as osp
import os

# set random seed for deterministic results
torch.manual_seed(1)
np.random.seed(1)

def random_sparsify(dataset, dataset_name, drop_rate):
    """
    Input:
        drop_rate: float, between 0 and 1
        size: int, number of elements to sparsify
        
    Output:
        edge_selection: tensor, shape (size), type torch.bool
    """
    myLogger.info(f'Getting prune file for {dataset_name}')
    path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/edge_selection.npy')
    if osp.exists(path):
        myLogger.info(message=f'Prune file already exists, loading edge selection')
        edge_selection = torch.load(path)
    else:
        myLogger.info(message=f'Prune file not exist, generating random prune file with drop rate {drop_rate}')
        if dataset_name in ['Reddit', 'Reddit2', 'ogbn_products']:
            size = dataset[0].edge_index.shape[1]
        edge_selection = torch.tensor(np.random.binomial(1, 1.0 - drop_rate, size=size)).type(torch.bool)
        os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save(edge_selection, path)
        myLogger.info(message=f'Prune file saved for future use')
    return edge_selection