import networkit as nk
import graph_tool.all as gt
import os
import os.path as osp
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm  
import scipy
from sklearn.cluster import SpectralClustering
import time, timeit

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ") 
    print("please source env.sh at the top level of the project")
    exit(1)

def readAllGraphsNK(dataset_name, config):
    graphs_dict = {}

    if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
    elif config[dataset_name]["directed"]:
        originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    elif config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    nk.graph.Graph.indexEdges(originalGraph)
    graphs_dict["original"] = [originalGraph]


    for prune_algo in ["LocalDegree", "LSpar", "GSpar", "LocalSimilarity", "SCAN"]:
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
            elif config[dataset_name]["directed"]:
                G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
            elif config[dataset_name]["weighted"]:
                G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
            else:
                G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
            G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
            graphs.append(G)
        graphs_dict[prune_algo] = graphs

    for prune_algo in ["Random", "KNeighbor", "RankDegree", "ForestFire"]:
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
                elif config[dataset_name]["directed"]:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
                elif config[dataset_name]["weighted"]:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
                else:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[prune_algo] = graphs
        
    for prune_algo in ["ER"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"]:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
                else:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_weighted"] = graphs

    for prune_algo in ["ER"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"]:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
                else:
                    G = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_unweighted"] = graphs

    for prune_algo in ["SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/uduw.el"
        if osp.exists(filepath):
            G = nk.readGraph(filepath, nk.Format.EdgeListSpaceZero, directed=False)
            G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
            graphs_dict[prune_algo] = [G]

    return graphs_dict


def readAllGraphsGT(dataset_name, config):
    graphs_dict = {}

    if config[dataset_name]["weighted"]:
        filepath = f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.wel"
        with open(filepath, "r") as f:
            el = np.loadtxt(f, dtype=float)
            originalGraph = gt.Graph(directed=True)
            eweight = originalGraph.new_edge_property("float")
            originalGraph.add_edge_list(el, eprops=[eweight])
            originalGraph.properties[("e", "weight")] = eweight
    else:
        filepath = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el"
        with open(filepath, "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            originalGraph = gt.Graph(directed=True)
            eweight = originalGraph.new_edge_property("float")
            originalGraph.add_edge_list(el, eprops=[eweight])
            originalGraph.properties[("e", "weight")] = eweight
    graphs_dict["original"] = [originalGraph]


    for prune_algo in ["LocalDegree", "LSpar", "GSpar", "LocalSimilarity", "SCAN"]:
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            if config[dataset_name]["weighted"]:
                filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=float)
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
            else:
                filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=int)
                    el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
            graphs.append(G)
        graphs_dict[prune_algo] = graphs

    for prune_algo in ["Random", "KNeighbor", "RankDegree", "ForestFire"]:
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["weighted"]:
                    filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel"
                    with open(filepath, "r") as f:
                        el = np.loadtxt(f, dtype=float)
                        G = gt.Graph(directed=True)
                        eweight = G.new_edge_property("float")
                        G.add_edge_list(el, eprops=[eweight])
                        G.properties[("e", "weight")] = eweight
                        G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                else:
                    filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el"
                    with open(filepath, "r") as f:
                        el = np.loadtxt(f, dtype=int)
                        el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                        G = gt.Graph(directed=True)
                        eweight = G.new_edge_property("float")
                        G.add_edge_list(el, eprops=[eweight])
                        G.properties[("e", "weight")] = eweight
                        G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                graphs.append(G)
        graphs_dict[prune_algo] = graphs
        
    for prune_algo in ["ER"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=float)
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_weighted"] = graphs

    for prune_algo in ["ER"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=int)
                    el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_unweighted"] = graphs


    for prune_algo in ["SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        filepath = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/duw.el"
        if osp.exists(filepath):
            with open(filepath, "r") as f:
                el = np.loadtxt(f, dtype=int)
                el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                G = gt.Graph(directed=True)
                eweight = G.new_edge_property("float")
                G.add_edge_list(el, eprops=[eweight])
                G.properties[("e", "weight")] = eweight
                G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
            graphs_dict[prune_algo] = [G]

    return graphs_dict