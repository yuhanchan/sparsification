import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.datasets as PygDataset


def Reddit():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = PygDataset.Reddit(path)
    return dataset

def Reddit2():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit2')
    dataset = PygDataset.Reddit2(path)
    return dataset

def ogbn_products():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = PygNodePropPredDataset(name='ogbn-products', root=path)
    data = dataset.data
    split_idx = dataset.get_idx_split()
    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f"{key}_mask"] = mask
    dataset.data = data
    return dataset