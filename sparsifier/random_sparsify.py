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
        dataset: PygDataset
        dataset_name: str, name of dataset
        drop_rate: float, between 0 and 1 
    Output:
        edge_selection: tensor, shape (size), type torch.bool
    """
    myLogger.info(f'Getting random prune file with prune rate {drop_rate} for {dataset_name}')
    prune_file_path = osp.join(osp.dirname(osp.abspath(__file__)), f'../data/{dataset_name}/pruned/random/{drop_rate}/edge_selection.npy')
    if osp.exists(prune_file_path):
        myLogger.info(message=f'Prune file already exists, loading edge selection')
        edge_selection = torch.load(prune_file_path)
    else:
        myLogger.info(message=f'Prune file not exist, generating random prune file with drop rate {drop_rate}')
        if dataset_name in ['Reddit', 'Reddit2', 'ogbn_products']:
            size = dataset[0].edge_index.shape[1]
        edge_selection = torch.tensor(np.random.binomial(1, 1.0 - drop_rate, size=size)).type(torch.bool)
        os.makedirs(osp.dirname(prune_file_path), exist_ok=True)
        torch.save(edge_selection, prune_file_path)
        myLogger.info(message=f'Prune file saved for future use')
    return edge_selection