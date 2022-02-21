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
    return dataset