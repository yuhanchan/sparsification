import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
import os.path as osp
import os
import numpy as np
from torch_sparse import SparseTensor
import pandas as pd
import sys


PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)


class Cheb_NET(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(Cheb_NET, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K=2, normalization=None))
        for _ in range(num_layers - 2):
            self.convs.append(
                ChebConv(hidden_channels, hidden_channels, K=2, normalization=None)
            )
        self.convs.append(ChebConv(hidden_channels, out_channels, K=2, normalization=None))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t, data.edge_weight)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t, data.edge_weight)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, valid_acc, test_acc


def Cheb_products(train_data, test_data, split_idx, evaluator, dataset_name, **kwargs):
    ## Default parameters
    # model parameters
    lr = 0.01
    dropout = 0.5
    epochs = 5000
    eval_each = 200
    num_layers = 3
    hidden_channels = 256
    runs = 1
    # run parameters
    experiment_dir = PROJECT_HOME + "/experiments/Cheb/" + dataset_name
    experiment_dir_postfix = ""
    load_ckpt = None
    load_pruned_graph = False
    do_inference_only = False
    inference_repeat = 10
    ckpt_interval = 10000  # set to a large number to disable checkpoint saving
    gpu_id = 0

    # Overwrite defaults
    lr = kwargs.get("lr", lr)
    dropout = kwargs.get("dropout", dropout)
    epochs = kwargs.get("epochs", epochs)
    eval_each = kwargs.get("eval_each", eval_each)
    num_layers = kwargs.get("num_layers", num_layers)
    hidden_channels = kwargs.get("hidden_channels", hidden_channels)
    runs = kwargs.get("runs", runs)
    experiment_dir = kwargs.get("experiment_dir", experiment_dir)
    experiment_dir_postfix = kwargs.get("experiment_dir_postfix", experiment_dir_postfix)
    load_ckpt = kwargs.get("load_ckpt", load_ckpt)
    load_pruned_graph = kwargs.get("load_pruned_graph", load_pruned_graph)
    do_inference_only = kwargs.get("do_inference_only", do_inference_only)
    inference_repeat = kwargs.get("inference_repeat", inference_repeat)
    ckpt_interval = kwargs.get("ckpt_interval", ckpt_interval)
    gpu_id = kwargs.get("gpu_id", gpu_id)

    # Sanity checks
    assert isinstance(lr, float)    
    assert isinstance(dropout, float)
    assert isinstance(epochs, int)
    assert isinstance(eval_each, int)
    assert isinstance(num_layers, int)
    assert isinstance(hidden_channels, int)
    assert isinstance(runs, int)
    assert isinstance(experiment_dir, str)
    assert isinstance(experiment_dir_postfix, str)
    assert isinstance(load_ckpt, str) or load_ckpt is None
    assert isinstance(load_pruned_graph, bool)
    assert isinstance(do_inference_only, bool)
    assert isinstance(inference_repeat, int)
    assert isinstance(ckpt_interval, int)
    assert isinstance(gpu_id, int)

    if (num_layers < 2) or (not isinstance(num_layers, int)):
        raise ValueError("num_layers should be integers >= 2")
    
    # Dump parameters
    experiment_dir = osp.join(experiment_dir, experiment_dir_postfix)
    if experiment_dir is not None:
        os.makedirs(experiment_dir, exist_ok=True)
        sys.stdout = open(f"{experiment_dir}/stdout.txt", "w")
        with open(os.path.join(experiment_dir, "args.txt"), "w") as f:
            f.write(f"---------------- params ----------------\n")
            f.write(f"lr = {lr}\n")
            f.write(f"dropout = {dropout}\n")
            f.write(f"epochs = {epochs}\n")
            f.write(f"eval_each = {eval_each}\n")
            f.write(f"num_layers = {num_layers}\n")
            f.write(f"hidden_channels = {hidden_channels}\n")
            f.write(f"runs = {runs}\n")
            f.write(f"experiment_dir = {experiment_dir}\n")
            f.write(f"load_ckpt = {load_ckpt}\n")
            f.write(f"load_pruned_graph = {load_pruned_graph}\n")
            f.write(f"do_inference_only = {do_inference_only}\n")
            f.write(f"inference_repeat = {inference_repeat}\n")
            f.write(f"ckpt_interval = {ckpt_interval}\n")
            f.write(f"gpu_id = {gpu_id}\n")
            # f.write(f"nnz in taining data: {train_data.adj_t.nnz}\n")
            f.write(f"----------------------------------------\n")

        
    if gpu_id == -1:
        device = "cpu"
    else:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_idx = split_idx["train"].to(device)

    model = Cheb_NET(
        train_data.num_features, hidden_channels, 47, num_layers, dropout
    ).to(device)

    train_data = train_data.to(device)
    test_data = test_data.to(device)

    train_scalars = []
    train_columns = ["Run", "Epoch", "Loss"]
    if experiment_dir is not None:
        pd.DataFrame(columns=train_columns).to_csv(
            osp.join(experiment_dir, "train.csv"), index=False
        )
    test_scalars = []
    test_columns = ["Run", "Epoch", "Loss", "Train ACC", "Valid ACC", "Test ACC"]
    if experiment_dir is not None:
        pd.DataFrame(columns=test_columns).to_csv(
            osp.join(experiment_dir, "test.csv"), index=False
        )

    logger = Logger(runs)

    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, 1 + epochs):
            loss = train(model, train_data, train_idx, optimizer)
            train_scalars.append([run, epoch, loss])
            if experiment_dir is not None:
                pd.DataFrame(data=train_scalars, columns=train_columns).to_csv(
                    osp.join(experiment_dir, "train.csv"),
                    mode="a",
                    index=False,
                    header=False,
                )
                train_scalars = []

            if epoch % eval_each == 0:
                result = test(model, test_data, split_idx, evaluator)
                logger.add_result(run, result)

                train_acc, valid_acc, test_acc = result
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )
                test_scalars.append([run, epoch, loss, train_acc, valid_acc, test_acc])
                if experiment_dir is not None:
                    pd.DataFrame(data=test_scalars, columns=test_columns).to_csv(
                        osp.join(experiment_dir, "test.csv"),
                        mode="a",
                        index=False,
                        header=False,
                    )
                    test_scalars = []

        logger.print_statistics(run)
    logger.print_statistics()

