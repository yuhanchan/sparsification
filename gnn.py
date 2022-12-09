import argparse
import sys
import copy

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_sparse
import numpy as np
from logger import Logger
import pickle
from scipy import sparse, stats
from numpy import inf
from torch_geometric.utils import from_scipy_sparse_matrix


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False)
            )
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


def ER_sampling(
    epsilon,
    Pe,
    C,
    weights,
    start_nodes,
    end_nodes,
    N,
):
    q = round(N * np.log(N) * 9 * C**2 / (epsilon**2))
    results = np.random.choice(np.arange(np.shape(Pe)[0]), int(q), p=list(Pe))
    spin_counts = stats.itemfreq(results).astype(int)

    per_spin_weights = weights / (q * Pe)
    per_spin_weights[per_spin_weights == inf] = 0

    counts = np.zeros(np.shape(weights)[0])
    counts[spin_counts[:, 0]] = spin_counts[:, 1]
    new_weights = counts * per_spin_weights

    sparserW = sparse.csc_matrix(
        (np.squeeze(new_weights), (start_nodes, end_nodes)), shape=(N, N)
    )

    sparserW = sparserW + sparserW.T

    print(
        f"Prune rate for epsilon={epsilon}: {1 - np.count_nonzero(new_weights) / np.size(new_weights)}, ({np.size(new_weights)} -> {np.count_nonzero(new_weights)})"
    )
    # convert into PyG's object
    edge_index, edge_weight = from_scipy_sparse_matrix(sparserW)

    return edge_index, edge_weight


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
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


def main():
    parser = argparse.ArgumentParser(description="OGBN-Products (GNN)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--use_sage", action="store_true")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pruned_file_path", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--reff_var_path", type=str, default="")
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--x_drop_rate", type=float, default=0)
    args = parser.parse_args()

    if args.log_path:
        sys.stdout = open(args.log_path, "w")

    print(args)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name="ogbn-products",
        transform=T.ToSparseTensor(),
        root="data",
    )
    data = dataset[0]

    if args.x_drop_rate > 0:
        # randomly remove some cols from 2d tensor data.x
        data.x = data.x[
            :,
            np.random.choice(
                np.arange(np.shape(data.x)[1]),
                int((1 - args.x_drop_rate) * np.shape(data.x)[1]),
                replace=False,
            ),
        ]

    # load pruned edge list
    if args.pruned_file_path != "":
        pruned_edge_list = np.loadtxt(
            args.pruned_file_path,
            dtype=int,
        )
        data.adj_t = torch_sparse.tensor.SparseTensor(
            row=torch.tensor(pruned_edge_list[:, 0], dtype=torch.long),
            col=torch.tensor(pruned_edge_list[:, 1], dtype=torch.long),
            sparse_sizes=data.adj_t.sizes(),
        )
        print(data)

    if args.reff_var_path != "":
        var = np.load("data/ogbn_products/raw/stage3.npz")
        Pe = var["Pe"]
        weights = var["weights"]
        start_nodes = var["start_nodes"]
        end_nodes = var["end_nodes"]
        N = var["N"]
        C0 = 1 / 30.0
        C = 4 * C0

    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)

    if args.use_sage:
        model = SAGE(
            data.num_features,
            args.hidden_channels,
            dataset.num_classes,
            args.num_layers,
            args.dropout,
        ).to(device)
    else:
        model = GCN(
            data.num_features,
            args.hidden_channels,
            dataset.num_classes,
            args.num_layers,
            args.dropout,
        ).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    # data = data.to(device)

    evaluator = Evaluator(name="ogbn-products")
    logger = Logger(args.runs, args)

    train_data = copy.deepcopy(data)
    # data = data.to(device)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            if args.reff_var_path != "" and epoch % 10 == 1:
                print("resampling edges")
                edge_index, edge_weight = ER_sampling(
                    args.epsilon,
                    Pe,
                    C,
                    weights,
                    start_nodes,
                    end_nodes,
                    N,
                )
                # edge_index = torch.vstack(
                #     (
                #         torch.cat(
                #             (
                #                 torch.tensor(start_nodes, dtype=torch.long),
                #                 torch.tensor(end_nodes, dtype=torch.long),
                #             ),
                #             dim=0,
                #         ),
                #         torch.cat(
                #             (
                #                 torch.tensor(end_nodes, dtype=torch.long),
                #                 torch.tensor(start_nodes, dtype=torch.long),
                #             ),
                #             dim=0,
                #         ),
                #     )
                # )

                train_data.adj_t = torch_sparse.tensor.SparseTensor(
                    row=edge_index[0],
                    col=edge_index[1],
                    sparse_sizes=data.adj_t.sizes(),
                )
                # train_data.adj_t = train_data.adj_t.set_diag()
                # train_data.adj_t = train_data.adj_t.set_value(
                # data.adj_t.storage.value()
                # )
                if not args.use_sage:
                    # Pre-compute GCN normalization.
                    adj_t = train_data.adj_t.set_diag()
                    deg = adj_t.sum(dim=1).to(torch.float)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
                    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
                    train_data.adj_t = adj_t
                print(train_data.adj_t)
                print(data.adj_t)

            data = data.to(device)
            train_data = train_data.to(device)
            loss = train(model, train_data, train_idx, optimizer)
            result = test(model, train_data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(
                    f"Run: {run + 1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {100 * train_acc:.2f}%, "
                    f"Valid: {100 * valid_acc:.2f}% "
                    f"Test: {100 * test_acc:.2f}%"
                )

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
