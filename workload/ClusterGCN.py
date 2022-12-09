import os
import sys
import time
import os.path as osp
from typing import Optional
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        hidden_channels = 256
        if (num_layers < 1) or (not isinstance(num_layers, int)):
            raise ValueError("num_layers should be integers >= 1")

        super(Net, self).__init__()
        self.inProj = torch.nn.Linear(in_channels, hidden_channels)
        if num_layers == 1:
            self.convs = ModuleList([SAGEConv(hidden_channels, out_channels, aggr="mean")])
        else:
            self.convs = ModuleList(
                [
                    SAGEConv(hidden_channels, hidden_channels, aggr="mean")
                    for _ in range(num_layers - 1)
                ]
                + [SAGEConv(hidden_channels, out_channels, aggr="mean")]
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.inProj(x)
        inp = x
        x = F.relu(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = x + 0.2 * inp
        return F.log_softmax(x, dim=-1)
    
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description("Evaluating")

        x_all = self.inProj(x_all.to(device))
        x_all = x_all.cpu()
        inp = x_all
        x_all = F.relu(x_all)

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_time = 0.0
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, original_edges, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                t = time.time()
                x = conv((x, x_target), edge_index) 
                torch.cuda.synchronize()
                total_time += time.time() - t
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
            if i != len(self.convs) - 1:
                x_all = x_all + 0.2 * inp
        pbar.close()
        print("inference time: ", total_time)
        return x_all


def train(model, train_loader, optimizer, device):
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        if (batch.y.shape[-1] != out.shape[-1]) and batch.y.shape[-1] == 1:
            y = batch.y.squeeze(1)[batch.train_mask]
            loss_ = F.nll_loss(out[batch.train_mask], y)
        else:
            loss_ = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss_.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss_.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(
    model,
    data,
    subgraph_loader,
    device,
    evaluator: Optional[Evaluator],
    repeat=1,
):  # Inference should be performed on the full graph.
    model.eval()
    for _ in range(repeat):
        out = model.inference(data.x, subgraph_loader, device)

    if evaluator is not None:
        y_pred = out.argmax(dim=-1, keepdim=True)
        train_accuracy = evaluator.eval(
            {
                "y_true": data.y[data.train_mask],
                "y_pred": y_pred[data.train_mask],
            }
        )["acc"]
        val_accuracy = evaluator.eval(
            {
                "y_true": data.y[data.valid_mask],
                "y_pred": y_pred[data.valid_mask],
            }
        )["acc"]
        test_accuracy = evaluator.eval(
            {
                "y_true": data.y[data.test_mask],
                "y_pred": y_pred[data.test_mask],
            }
        )["acc"]
        accs = [train_accuracy, val_accuracy, test_accuracy]
    else:
        y_pred = out.argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = y_pred[mask].eq(data.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
    return accs


def ClusterGCN(train_dataset, test_dataset, dataset_name, **kwargs):
    # Default parameters
    experiment_dir = None
    epochs = 30
    eval_each = 5
    num_layers = 4
    init_checkpoint = None
    load_pruned_graph = False
    do_inference_only = False
    inference_repeat = 10
    inference_batch_size = 40000
    save_ckpt_space = False
    gpu_id = 0

    # Overwrite defaults
    experiment_dir = kwargs.get("experiment_dir", experiment_dir)
    epochs = kwargs.get("epochs", epochs)
    eval_each = kwargs.get("eval_each", eval_each)
    num_layers = kwargs.get("num_layers", num_layers)
    init_checkpoint = kwargs.get("init_checkpoint", init_checkpoint)
    load_pruned_graph = kwargs.get("load_pruned_graph", load_pruned_graph)
    do_inference_only = kwargs.get("do_inference_only", do_inference_only)
    inference_repeat = kwargs.get("inference_repeat", inference_repeat)
    inference_batch_size = kwargs.get("inference_batch_size", inference_batch_size)
    save_ckpt_space = kwargs.get("save_ckpt_space", save_ckpt_space)
    gpu_id = kwargs.get("gpu_id", gpu_id)

    # Sanity checks
    assert isinstance(experiment_dir, str)
    assert isinstance(epochs, int)
    assert isinstance(eval_each, int)
    assert isinstance(num_layers, int)
    assert isinstance(init_checkpoint, str)
    assert isinstance(load_pruned_graph, bool)
    assert isinstance(do_inference_only, bool)
    assert isinstance(inference_repeat, int)
    assert isinstance(inference_batch_size, int)
    assert isinstance(save_ckpt_space, bool)
    assert isinstance(gpu_id, int)

    if (num_layers < 1) or (not isinstance(num_layers, int)):
        raise ValueError("num_layers should be integers >= 1")

    # Dump parameters
    if experiment_dir is not None:
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "args.txt"), "w") as f:
            f.write(f"expriment_dir = {experiment_dir}")
            f.write(f"epochs = {epochs}")
            f.write(f"eval_each = {eval_each}")
            f.write(f"num_layers = {num_layers}")
            f.write(f"init_checkpoint = {init_checkpoint}")
            f.write(f"load_pruned_graph = {load_pruned_graph}")
            f.write(f"do_inference_only = {do_inference_only}")
            f.write(f"inference_repeat = {inference_repeat}")
            f.write(f"inference_batch_size = {inference_batch_size}")
            f.write(f"save_ckpt_space = {save_ckpt_space}")
            f.write(f"gpu_id = {gpu_id}")

    # Dataset
    if dataset_name == "Reddit":
        evaluator = None
        train_data = train_dataset.data
        test_data = test_dataset.data
    elif dataset_name == "Reddit2":
        evaluator = None
        train_data = train_dataset.data
        test_data = test_dataset.data
    elif dataset_name == "ogbn-products":
        evaluator = Evaluator(name=dataset_name)
        train_data = train_dataset.data
        test_data = test_dataset.data
    else:
        print("Dataset not supported.")
        sys.exit(1)

    # Model
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = Net(train_dataset.num_features, train_dataset.num_classes, num_layers).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initial checkpoint
    if init_checkpoint is not None:
        ckp = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(ckp["model"])
        if load_pruned_graph:
            data = ckp["data"]

    if not do_inference_only:
        cluster_data = ClusterData(
            train_data,
            num_parts=15000,
            recursive=False,  # , save_dir=dataset.processed_dir
        )
        train_loader = ClusterLoader(
            cluster_data, batch_size=32, shuffle=True, num_workers=1
        )

    subgraph_loader = NeighborSampler(
        test_data.edge_index,
        sizes=[-1],
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=1,
    )

    # Inference only
    if do_inference_only:
        train_acc, val_acc, test_acc = test(
            model,
            test_data,
            subgraph_loader,
            device,
            evaluator,
            repeat=inference_repeat,
        )
        print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        exit(0)

    if experiment_dir is not None:
        ckp = {"model": model.state_dict(), "data": data}
        torch.save(ckp, osp.join(experiment_dir, "init_ckp.pth.tar"))

    train_scalars = []
    train_columns = ["Epoch", "Loss"]
    if experiment_dir is not None:
        pd.DataFrame(columns=train_columns).to_csv(
            osp.join(experiment_dir, "train.csv"), index=False
        )
    test_scalars = []
    test_columns = ["Epoch", "Loss", "Train Acc", "Val Acc", "Test Acc"]
    if experiment_dir is not None:
        pd.DataFrame(columns=test_columns).to_csv(
            osp.join(experiment_dir, "test.csv"), index=False
        )

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
        train_scalars.append([epoch, loss])
        if experiment_dir is not None:
            pd.DataFrame(data=train_scalars, columns=train_columns).to_csv(
                osp.join(experiment_dir, "train.csv"),
                mode="a",
                index=False,
                header=False,
            )
            train_scalars = []
            # if save_ckpt_space:
            #     if epoch == epochs:
            #         ckp = {"model": model.state_dict(), "data": data}
            #         torch.save(
            #             ckp,
            #             osp.join(
            #                 experiment_dir, "ckp_epoch_" + str(epoch) + ".pth.tar"
            #             ),
            #         )
            # elif epoch % 50 == 0:
            # ckp = {"model": model.state_dict(), "data": data}
            # torch.save(
            #     ckp,
            #     osp.join(experiment_dir, "ckp_epoch_" + str(epoch) + ".pth.tar"),
            # )

        if epoch % eval_each == 0:
            train_acc, val_acc, test_acc = test(
                model, test_data, subgraph_loader, device, evaluator
            )
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )
            test_scalars.append([epoch, loss, train_acc, val_acc, test_acc])
            if experiment_dir is not None:
                pd.DataFrame(data=test_scalars, columns=test_columns).to_csv(
                    osp.join(experiment_dir, "test.csv"),
                    mode="a",
                    index=False,
                    header=False,
                )
                test_scalars = []
