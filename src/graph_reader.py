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

def readAllGraphsNK(dataset_name, config):
    graphs_dict = {}

    if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
    elif config[dataset_name]["directed"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    elif config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    nk.graph.Graph.indexEdges(originalGraph)
    graphs_dict["original"] = [originalGraph]


    for prune_algo in ["LocalDegree", "LSpar", "GSpar", "LocalSimilarity", "SCAN"]:
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
            elif config[dataset_name]["directed"]:
                G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
            elif config[dataset_name]["weighted"]:
                G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
            else:
                G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
            G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
            graphs.append(G)
        graphs_dict[prune_algo] = graphs

    for prune_algo in ["Random", "KNeighbor", "RankDegree", "ForestFire"]:
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
                elif config[dataset_name]["directed"]:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
                elif config[dataset_name]["weighted"]:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
                else:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[prune_algo] = graphs
        
    for prune_algo in ["ER-Min", "ER-Max"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"]:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
                else:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_weighted"] = graphs

    for prune_algo in ["ER-Min", "ER-Max"]: # unweighted er
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"]:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
                else:
                    G = nk.readGraph(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
                G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_unweighted"] = graphs

    for prune_algo in ["SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        filepath = f"data/{dataset_name}/pruned/{prune_algo}/0/uduw.el"
        if osp.exists(filepath):
            G = nk.readGraph(filepath, nk.Format.EdgeListSpaceZero, directed=False)
            G.addNodes(originalGraph.numberOfNodes()-G.numberOfNodes())
            graphs_dict[prune_algo] = [G]

    return graphs_dict


def readAllGraphsGT(dataset_name, config):
    graphs_dict = {}

    if config[dataset_name]["weighted"]:
        filepath = f"data/{dataset_name}/raw/dw.wel"
        with open(filepath, "r") as f:
            el = np.loadtxt(f, dtype=float)
            originalGraph = gt.Graph(directed=True)
            eweight = originalGraph.new_edge_property("float")
            originalGraph.add_edge_list(el, eprops=[eweight])
            originalGraph.properties[("e", "weight")] = eweight
    else:
        filepath = f"data/{dataset_name}/raw/duw.el"
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
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            if config[dataset_name]["weighted"]:
                filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=float)
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
            else:
                filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el"
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
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["weighted"]:
                    filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel"
                    with open(filepath, "r") as f:
                        el = np.loadtxt(f, dtype=float)
                        G = gt.Graph(directed=True)
                        eweight = G.new_edge_property("float")
                        G.add_edge_list(el, eprops=[eweight])
                        G.properties[("e", "weight")] = eweight
                        G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                else:
                    filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el"
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
        
    for prune_algo in ["ER-Min", "ER-Max"]: # weighted er
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel"
                with open(filepath, "r") as f:
                    el = np.loadtxt(f, dtype=float)
                    G = gt.Graph(directed=True)
                    eweight = G.new_edge_property("float")
                    G.add_edge_list(el, eprops=[eweight])
                    G.properties[("e", "weight")] = eweight
                    G.add_vertex(originalGraph.num_vertices()-G.num_vertices())
                graphs.append(G)
        graphs_dict[f"{prune_algo}_weighted"] = graphs

    for prune_algo in ["ER-Min", "ER-Max"]: # unweighted er
        graphs = []
        for prune_rate in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                filepath = f"data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el"
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
        filepath = f"data/{dataset_name}/pruned/{prune_algo}/0/duw.el"
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


def readGraphs(dataset_name, targetRatio, config, readGTformat=False, er_weighted=True):
    global originalGraph, erMinGraph, erMaxGraph, spanningGraph, fireGraph, localDegGraph, localSimilarityGraph, randomGraph, scanGraph, simmelieanGraph, jaccardGraph, spanner_3, spanner_5, spanner_7
    global originalGraph_gt, erMinGraph_gt, erMaxGraph_gt, spanningGraph_gt, fireGraph_gt, localDegGraph_gt, localSimilarityGraph_gt, randomGraph_gt, scanGraph_gt, simmelieanGraph_gt, jaccardGraph_gt, spanner_3_gt, spanner_5_gt, spanner_7_gt

    if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True, weighted=True)
    elif config[dataset_name]["directed"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True, weighted=False)
    elif config[dataset_name]["weighted"]:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False, weighted=True)
    else:
        originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False, weighted=False)
    nk.graph.Graph.indexEdges(originalGraph)

    if readGTformat:
        if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
            filepath = f"data/{dataset_name}/raw/dw.wel"
        elif config[dataset_name]["directed"]:
            filepath = f"data/{dataset_name}/raw/duw.el"
        elif config[dataset_name]["weighted"]:
            filepath = f"data/{dataset_name}/raw/udw.wel"
        else:
            filepath = f"data/{dataset_name}/raw/uduw.el"
        with open(f"data/{dataset_name}/raw/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            # el = np.vstack((el, np.vstack((el[:,1], el[:,0])).T)) # symmetrize
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            originalGraph_gt = gt.Graph(directed=True)
            eweight = originalGraph_gt.new_edge_property("float")
            originalGraph_gt.add_edge_list(el, eprops=[eweight])
            originalGraph_gt.properties[("e", "weight")] = eweight


    if er_weighted:
        if config[dataset_name]["directed"]:
            erMinGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
        else:
            erMinGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        if config[dataset_name]["directed"]:
            erMinGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        else:
            erMinGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    erMinGraph.addNodes(originalGraph.numberOfNodes()-erMinGraph.numberOfNodes())

    if readGTformat:
        if er_weighted:
            with open(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/dw.wel", "r") as f:
                el = np.loadtxt(f, dtype=float)
                erMinGraph_gt = gt.Graph(directed=True)
                eweight = erMinGraph_gt.new_edge_property("float")
                erMinGraph_gt.add_edge_list(el, eprops=[eweight])
                erMinGraph_gt.add_vertex(originalGraph_gt.num_vertices()-erMinGraph_gt.num_vertices())
                erMinGraph_gt.properties[("e", "weight")] = eweight
        else:
            with open(f"data/{dataset_name}/pruned/er_min/{round(1-targetRatio, 3)}/duw.el", "r") as f:
                el = np.loadtxt(f, dtype=int)
                # add a column of 1s for weights
                el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                erMinGraph_gt = gt.Graph(directed=True)
                eweight = erMinGraph_gt.new_edge_property("float")
                erMinGraph_gt.add_edge_list(el, eprops=[eweight])
                erMinGraph_gt.add_vertex(originalGraph_gt.num_vertices()-erMinGraph_gt.num_vertices())
                erMinGraph_gt.properties[("e", "weight")] = eweight

    if er_weighted:
        if config[dataset_name]["directed"]:
            erMaxGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
        else:
            erMaxGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        if config[dataset_name]["directed"]:
            erMaxGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        else:
            erMaxGraph = nk.readGraph(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    erMaxGraph.addNodes(originalGraph.numberOfNodes()-erMaxGraph.numberOfNodes())

    if readGTformat:
        if er_weighted:
            with open(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/dw.wel", "r") as f:
                el = np.loadtxt(f, dtype=float)
                erMaxGraph_gt = gt.Graph(directed=True)
                eweight = erMaxGraph_gt.new_edge_property("float")
                erMaxGraph_gt.add_edge_list(el, eprops=[eweight])
                erMaxGraph_gt.add_vertex(originalGraph_gt.num_vertices()-erMaxGraph_gt.num_vertices())
                erMaxGraph_gt.properties[("e", "weight")] = eweight
        else:
            with open(f"data/{dataset_name}/pruned/er_max/{round(1-targetRatio, 3)}/duw.el", "r") as f:
                el = np.loadtxt(f, dtype=int)
                # add a column of 1s for weights
                el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
                erMaxGraph_gt = gt.Graph(directed=True)
                eweight = erMaxGraph_gt.new_edge_property("float")
                erMaxGraph_gt.add_edge_list(el, eprops=[eweight])
                erMaxGraph_gt.add_vertex(originalGraph_gt.num_vertices()-erMaxGraph_gt.num_vertices())
                erMaxGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        fireGraph = nk.readGraph(f"data/{dataset_name}/pruned/forestfire/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        fireGraph = nk.readGraph(f"data/{dataset_name}/pruned/forestfire/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    fireGraph.addNodes(originalGraph.numberOfNodes()-fireGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/forestfire/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            fireGraph_gt = gt.Graph(directed=True)
            eweight = fireGraph_gt.new_edge_property("float")
            fireGraph_gt.add_edge_list(el, eprops=[eweight])
            fireGraph_gt.add_vertex(originalGraph_gt.num_vertices()-fireGraph_gt.num_vertices())
            fireGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        localDegGraph = nk.readGraph(f"data/{dataset_name}/pruned/localdeg/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        localDegGraph = nk.readGraph(f"data/{dataset_name}/pruned/localdeg/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    localDegGraph.addNodes(originalGraph.numberOfNodes()-localDegGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/localdeg/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            localDegGraph_gt = gt.Graph(directed=True)
            eweight = localDegGraph_gt.new_edge_property("float")
            localDegGraph_gt.add_edge_list(el, eprops=[eweight])
            localDegGraph_gt.add_vertex(originalGraph_gt.num_vertices()-localDegGraph_gt.num_vertices())
            localDegGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        localSimilarityGraph = nk.readGraph(f"data/{dataset_name}/pruned/localSimilarity/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        localSimilarityGraph = nk.readGraph(f"data/{dataset_name}/pruned/localSimilarity/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    localSimilarityGraph.addNodes(originalGraph.numberOfNodes()-localSimilarityGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/localSimilarity/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            localSimilarityGraph_gt = gt.Graph(directed=True)
            eweight = localSimilarityGraph_gt.new_edge_property("float")
            localSimilarityGraph_gt.add_edge_list(el, eprops=[eweight])
            localSimilarityGraph_gt.add_vertex(originalGraph_gt.num_vertices()-localSimilarityGraph_gt.num_vertices())
            localSimilarityGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        randomGraph = nk.readGraph(f"data/{dataset_name}/pruned/random/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        randomGraph = nk.readGraph(f"data/{dataset_name}/pruned/random/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    randomGraph.addNodes(originalGraph.numberOfNodes()-randomGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/random/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            randomGraph_gt = gt.Graph(directed=True)
            eweight = randomGraph_gt.new_edge_property("float")
            randomGraph_gt.add_edge_list(el, eprops=[eweight])
            randomGraph_gt.add_vertex(originalGraph_gt.num_vertices()-randomGraph_gt.num_vertices())
            randomGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        scanGraph = nk.readGraph(f"data/{dataset_name}/pruned/scan/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        scanGraph = nk.readGraph(f"data/{dataset_name}/pruned/scan/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    scanGraph.addNodes(originalGraph.numberOfNodes()-scanGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/scan/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            scanGraph_gt = gt.Graph(directed=True)
            eweight = scanGraph_gt.new_edge_property("float")
            scanGraph_gt.add_edge_list(el, eprops=[eweight])
            scanGraph_gt.add_vertex(originalGraph_gt.num_vertices()-scanGraph_gt.num_vertices())
            scanGraph_gt.properties[("e", "weight")] = eweight


    # if config[dataset_name]["directed"]:
    #     simmelieanGraph = nk.readGraph(f"data/{dataset_name}/pruned/simmelian/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    # else:
    #     simmelieanGraph = nk.readGraph(f"data/{dataset_name}/pruned/simmelian/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    # simmelieanGraph.addNodes(originalGraph.numberOfNodes()-simmelieanGraph.numberOfNodes())
    # if readGTformat:
    #     with open(f"data/{dataset_name}/pruned/simmelian/{round(1-targetRatio, 3)}/duw.el", "r") as f:
    #         el = np.loadtxt(f, dtype=int)
    #         el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
    #         simmelieanGraph_gt = gt.Graph(directed=True)
    #         eweight = simmelieanGraph_gt.new_edge_property("float")
    #         simmelieanGraph_gt.add_edge_list(el, eprops=[eweight])
    #         simmelieanGraph_gt.add_vertex(originalGraph_gt.num_vertices()-simmelieanGraph_gt.num_vertices())
    #         simmelieanGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        jaccardGraph = nk.readGraph(f"data/{dataset_name}/pruned/jaccard/{round(1-targetRatio, 3)}/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        jaccardGraph = nk.readGraph(f"data/{dataset_name}/pruned/jaccard/{round(1-targetRatio, 3)}/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    jaccardGraph.addNodes(originalGraph.numberOfNodes()-jaccardGraph.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/jaccard/{round(1-targetRatio, 3)}/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            jaccardGraph_gt = gt.Graph(directed=True)
            eweight = jaccardGraph_gt.new_edge_property("float")
            jaccardGraph_gt.add_edge_list(el, eprops=[eweight])
            jaccardGraph_gt.add_vertex(originalGraph_gt.num_vertices()-jaccardGraph_gt.num_vertices())
            jaccardGraph_gt.properties[("e", "weight")] = eweight


    if config[dataset_name]["directed"]:
        spanningGraph = nk.readGraph(f"data/{dataset_name}/pruned/spanning/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        spanner_3 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-3/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        spanner_5 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-5/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        spanner_7 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-7/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
    else:
        spanningGraph = nk.readGraph(f"data/{dataset_name}/pruned/spanning/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
        spanner_3 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-3/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
        spanner_5 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-5/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
        spanner_7 = nk.readGraph(f"data/{dataset_name}/pruned/spanner-7/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    spanningGraph.addNodes(originalGraph.numberOfNodes()-spanningGraph.numberOfNodes())
    spanner_3.addNodes(originalGraph.numberOfNodes()-spanner_3.numberOfNodes())
    spanner_5.addNodes(originalGraph.numberOfNodes()-spanner_5.numberOfNodes())
    spanner_7.addNodes(originalGraph.numberOfNodes()-spanner_7.numberOfNodes())
    if readGTformat:
        with open(f"data/{dataset_name}/pruned/spanning/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            spanningGraph_gt = gt.Graph(directed=True)
            eweight = spanningGraph_gt.new_edge_property("float")
            spanningGraph_gt.add_edge_list(el, eprops=[eweight])
            spanningGraph_gt.add_vertex(originalGraph_gt.num_vertices()-spanningGraph_gt.num_vertices())
            spanningGraph_gt.properties[("e", "weight")] = eweight
        with open(f"data/{dataset_name}/pruned/spanner-3/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            spanner_3_gt = gt.Graph(directed=True)
            eweight = spanner_3_gt.new_edge_property("float")
            spanner_3_gt.add_edge_list(el, eprops=[eweight])
            spanner_3_gt.add_vertex(originalGraph_gt.num_vertices()-spanner_3_gt.num_vertices())
            spanner_3_gt.properties[("e", "weight")] = eweight
        with open(f"data/{dataset_name}/pruned/spanner-5/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            spanner_5_gt = gt.Graph(directed=True)
            eweight = spanner_5_gt.new_edge_property("float")
            spanner_5_gt.add_edge_list(el, eprops=[eweight])
            spanner_5_gt.add_vertex(originalGraph_gt.num_vertices()-spanner_5_gt.num_vertices())
            spanner_5_gt.properties[("e", "weight")] = eweight
        with open(f"data/{dataset_name}/pruned/spanner-7/duw.el", "r") as f:
            el = np.loadtxt(f, dtype=int)
            el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))
            spanner_7_gt = gt.Graph(directed=True)
            eweight = spanner_7_gt.new_edge_property("float")
            spanner_7_gt.add_edge_list(el, eprops=[eweight])
            spanner_7_gt.add_vertex(originalGraph_gt.num_vertices()-spanner_7_gt.num_vertices())
            spanner_7_gt.properties[("e", "weight")] = eweight

    G = {"originalGraph       ": originalGraph, 
         "erMinGraph          ": erMinGraph, 
         "erMaxGraph          ": erMaxGraph, 
         "spanningGraph       ": spanningGraph, 
         "spanner_3           ": spanner_3,
         "spanner_5           ": spanner_5,
         "spanner_7           ": spanner_7,
         "fireGraph           ": fireGraph, 
         "localDegGraph       ": localDegGraph, 
         "localSimilarityGraph": localSimilarityGraph, 
         "randomGraph         ": randomGraph, 
         "scanGraph           ": scanGraph, 
        #  "simmelieanGraph     ": simmelieanGraph, 
         "jaccardGraph        ": jaccardGraph}

    G_gt = {"originalGraph       ": originalGraph_gt, 
            "erMinGraph          ": erMinGraph_gt, 
            "erMaxGraph          ": erMaxGraph_gt, 
            "spanningGraph       ": spanningGraph_gt, 
            "spanner_3           ": spanner_3_gt,
            "spanner_5           ": spanner_5_gt,
            "spanner_7           ": spanner_7_gt,
            "fireGraph           ": fireGraph_gt, 
            "localDegGraph       ": localDegGraph_gt, 
            "localSimilarityGraph": localSimilarityGraph_gt, 
            "randomGraph         ": randomGraph_gt, 
            "scanGraph           ": scanGraph_gt, 
            # "simmelieanGraph     ": simmelieanGraph_gt, 
            "jaccardGraph        ": jaccardGraph_gt}

    return G, G_gt
