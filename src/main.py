import networkit as nk
import graph_tool.all as gt
import os
import os.path as osp
import random
from itertools import chain
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm  
import scipy
from sklearn.cluster import SpectralClustering
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from multiprocessing import set_start_method
import time, timeit
import sys
import psutil
import json
from metrics_nk import *
from metrics_gt import *
from sparsifiers import *
from datasets import *
from graph_reader import *
from memory_profiler import memory_usage
import argparse
 


def graphOverview(G):
    for name, Graph in G.items():
        nk.overview(Graph)

def loadOriginalGraph(dataset_name, config, undirected_only=False):
    if undirected_only:
        if config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
        else:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["directed"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
        else:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)

    nk.overview(originalGraph)
    nk.graph.Graph.indexEdges(originalGraph)
    return originalGraph


def graphSparsifier(dataset_name, targetRatios, config, mode, multi_process=False, num_run=1):
    """This function generates sparsified graphs with different sparsification algorithms, and save to files

    Args:
        dataset_name (str): dataset name
    """

    # print overview only
    if mode == 0:
        if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["directed"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
        else:
            originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
        nk.overview(originalGraph)

    # ER min and ER max
    elif mode == 1:
        print("ER min:")
        for i in range(num_run):
            ERMinSparsifier(dataset_name, config, multi_process=True, postfix_folder=str(i))
        print("ER max:")
        for i in range(num_run):
            ERMaxSparsifier(dataset_name, config, multi_process=True, postfix_folder=str(i))

    # random, local degree, forestfire, local similarity, GSpar, scan, RankDegree
    elif mode == 2:
        originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=False)
        for targetRatio in targetRatios:
            targetRatio = round(targetRatio, 3)
            if multi_process:
                with ProcessPoolExecutor(max_workers=128) as executor:
                    for i in range(num_run):
                        executor.submit(RandomEdgeSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                        executor.submit(ForestFireSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                        executor.submit(RankDegreeSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                    # following are same for all runs
                    executor.submit(LocalDegreeSparsifier, dataset_name, originalGraph, targetRatio, config, "0")
                    executor.submit(GSpar, dataset_name, originalGraph, targetRatio, config, "0")
                    executor.submit(LocalSimilaritySparsifier, dataset_name, originalGraph, targetRatio, config, "0")
                    executor.submit(SCANSparsifier, dataset_name, originalGraph, targetRatio, config, "0")
            else:
                for i in range(num_run):
                    RandomEdgeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))
                    ForestFireSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))
                    RankDegreeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))
                # following are same for all runs
                LocalDegreeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")
                GSpar(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")
                LocalSimilaritySparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")
                SCANSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")

    # KNeighbor
    elif mode == 3:
        originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=False)
        for k in config[dataset_name]["kNeighbors"]:
            for i in range(num_run):
                if multi_process:
                    with ProcessPoolExecutor(max_workers=128) as executor:
                        executor.submit(KNeighborSparsifier, dataset_name, originalGraph, k, config, str(i))
                else:
                    KNeighborSparsifier(dataset_name, originalGraph, k, config, postfix_folder=str(i))

    # t-spanner, always use undirected graph, same across multiple runs, so run only once
    elif mode == 4:
        with ProcessPoolExecutor(max_workers=128) as executor:
            futures = []
            if type(dataset_name) is not list:
                originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=True)
                for t in [3, 5, 7]:
                    futures.append(executor.submit(wrapped_spanner, dataset_name, originalGraph, t, config, "0"))
            else:
                for d in dataset_name:
                    originalGraph = loadOriginalGraph(d, config, undirected_only=True)
                    for t in [3, 5, 7]:
                        futures.append(executor.submit(wrapped_spanner, d, originalGraph, t, config, "0"))
            for future in futures:
                future.result()

    # spanning, always use undirected graph
    elif mode == 5: 
        originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=True)
        # same across multiple runs, so run only once
        SpanningForestSparsifier(dataset_name, originalGraph, config, postfix_folder="0")

    # LSpar
    elif mode == 6:
        originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=False)
        for c in config[dataset_name]["LSpar_c"]: # always the same across runs
            if multi_process:
                with ProcessPoolExecutor(max_workers=128) as executor:
                    executor.submit(LSpar, dataset_name, originalGraph, config, c, "0")
            else:
                LSpar(dataset_name, originalGraph, config, c, postfix_folder="0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="name of the dataset, or 'all'")
    parser.add_argument("--mode", choices=["sparsify", "eval", "all"], required=True, help="sparsify: only sparsify graphs; eval: only evaluate metrics (need to have run sparsify first); all: run both sparsify and eval")
    args = parser.parse_args()

    config = json.load(open("config.json", "r"))

    if args.dataset_name == "all":
        dataset_names = ["ego-Facebook", 
                        "ego-Twitter", 
                        "human_gene2", 
                        "com-DBLP", 
                        "com-Amazon", 
                        "email-Enron", 
                        "wiki-Talk", 
                        "ca-AstroPh", 
                        "ca-HepPh", 
                        "web-BerkStan", 
                        "web-Google", 
                        "web-NotreDame", 
                        "web-Stanford", 
                        "Reddit", 
                        "ogbn-proteins"
                        ]
    else:
        dataset_names = [args.dataset_name]


    # # mode 0: print overview
    # for dataset_name in dataset_names:
    #     print(dataset_name)
    #     graphSparsifier(dataset_name=dataset_name, targetRatios=[], config=config, mode=0)
    #     print("-----------------------")

    if args.mode == "sparsify" or args.mode == "all":
        # mode 1: er min and er max
        for dataset_name in dataset_names:
            # print to terminal
            sys.stdout = sys.__stdout__
            print("\n\n----------------------------------")
            print(dataset_name)
            # redirect print to file, append mode
            # sys.stdout = open(f"output/{dataset_name}.txt", "a")
            graphSparsifier(dataset_name=dataset_name, targetRatios=[], config=config, mode=1)

        # mode 2: random, localDegree, forestfire, Gspar, localsim, SCAN, RankDegree
        for dataset_name in dataset_names:
            # print to terminal
            sys.stdout = sys.__stdout__
            print("\n\n----------------------------------")
            print(dataset_name)
            # redirect print to file, append mode
            # sys.stdout = open(f"output/{dataset_name}.txt", "a")
            prune_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99] 
            graphSparsifier(dataset_name=dataset_name, targetRatios=[1-x for x in prune_rate], config=config, mode=2, multi_process=True)

        # mode 3: KNeighbor
        for dataset_name in dataset_names:
            # print to terminal
            sys.stdout = sys.__stdout__
            print("\n\n----------------------------------")
            print(dataset_name)
            # redirect print to file, append mode
            # sys.stdout = open(f"output/{dataset_name}.txt", "a")
            print()
            graphSparsifier(dataset_name=dataset_name, targetRatios=[], config=config, mode=3)

        # mode 4: t-spanner
        graphSparsifier(dataset_name=dataset_names, targetRatios=[], config=config, mode=4)

        # mode 5: spanningForest
        for dataset_name in dataset_names:
            # print to terminal
            sys.stdout = sys.__stdout__
            print("\n\n----------------------------------")
            print(dataset_name)
            # redirect print to file, append mode
            # sys.stdout = open(f"output/{dataset_name}.txt", "a")
            graphSparsifier(dataset_name=dataset_name, targetRatios=[], config=config, mode=5)

        # mode 6: LSpar
        for dataset_name in dataset_names:
            # print to terminal
            sys.stdout = sys.__stdout__
            print("\n\n----------------------------------")
            print(dataset_name)
            # redirect print to file, append mode
            # sys.stdout = open(f"output/{dataset_name}.txt", "a")
            print()
            graphSparsifier(dataset_name=dataset_name, targetRatios=[], config=config, mode=6)


    if args.mode == "eval" or args.mode == "all":
        for dataset_name in dataset_names:
            G_nk_dict = readAllGraphsNK(dataset_name, config)
            degreeDistribution_nk(dataset_name, G_nk_dict, nbin=100, logToFile=True)
            Centrality_nk(dataset_name, "EstimateBetweenness", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "TopCloseness", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "Degree", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "Katz", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "Laplacian", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "Eigenvector", G_nk_dict, 100, logToFile=True)
            Centrality_nk(dataset_name, "CoreDecomposition", G_nk_dict, 100, logToFile=True)
            DetectCommunity_nk(dataset_name, G_nk_dict, logToFile=True)
            ClusteringF1Similarity_nk(dataset_name, G_nk_dict, logToFile=True)
            Centrality_nk(dataset_name, "PageRank", G_nk_dict, 100, logToFile=False)
            QuadraticFormSimilarity_nk(dataset_name, G_nk_dict, logToFile=True)
            del G_nk_dict # release memory

            G_gt_dict = readAllGraphsGT(dataset_name, config) 
            ApproximateDiameter_gt(dataset_name, G_gt_dict, logToFile=True)
            SPSP_Eccentricity_gt(dataset_name, G_gt_dict, num_nodes=100, logToFile=True)
            GlobalClusteringCoefficient_gt(dataset_name, G_gt_dict, logToFile=True)
            LocalClusteringCoefficient_gt(dataset_name, G_gt_dict, logToFile=True)
            MaxFlow_gt(dataset_name, G_gt_dict, logToFile=True)
            del G_gt_dict # release memory


if __name__ == "__main__":
    set_start_method("spawn") # must write this line under if __name__ == "__main__":, otherwise it will not work
    main() 


""" unused code

    G_nk_dict["er_min_weighted"] = []
    G_nk_dict["er_max_weighted"] = []
    for name, Graphs in G_nk_dict.items():
        for Graph in Graphs:
            diameter = nk.distance.Diameter(Graph, algo=nk.distance.DiameterAlgo.Exact).run().getDiameter()
            print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()}\t diameter: {diameter}")

    nk.overview(G_nk_dict["original"][0])
    baseline = np.array(page_rank(G_nk_dict["original"][0], max_iter=1000)).flatten()
    for name, Graphs in G_nk_dict.items():
        for Graph in Graphs:
            pr = np.array(page_rank(Graph, max_iter=1000)).flatten()
            print(name, 1-Graph.numberOfEdges()/G_nk_dict["original"][0].numberOfEdges(), ranking_precision(baseline, pr, k=100))

    # ClusteringF1SimilarityWithGroundTruth_nk(dataset_name, G_nk_dict, osp.join(PROJECT_HOME, "data/com-Amazon/raw/com-amazon.all.dedup.cmty.remap.txt"), logToFile=False)

    # # # basic
    # degreeDistribution_nk(G_nk_dict)

    # # # disatance, road network
    # # EffectiveDiameter_nk(G_nk_dict)
    # # Eccentricity_nk(G_nk_dict, num_nodes=10000)
    # # SPSP_nk(G_nk_dict, num_nodes=100)

    # # # Centrality, social network
    # # Centrality_nk("Betweenness", G_nk_dict, 100, Approximate=False)
    # Centrality_nk("EstimateBetweenness", G_nk_dict, 100)
    # # Centrality_nk("DynBetweenness", G_nk_dict, 100, Approximate=True)
    # # Centrality_nk("Closeness", G_nk_dict, 100, Approximate=False)
    # Centrality_nk("Closeness", G_nk_dict, 100, Approximate=True)
    # Centrality_nk("Degree", G_nk_dict, 100)
    # Centrality_nk("KPath", G_nk_dict, 100)
    # Centrality_nk("Katz", G_nk_dict, 100)
    # Centrality_nk("Laplacian", G_nk_dict, 100)
    # # Centrality_nk("LocalClusteringCoefficient", G_nk_dict, 100)
    # Centrality_nk("Eigenvector", G_nk_dict, 100)
    # Centrality_nk("CoreDecomposition", G_nk_dict, 100)

    # # # Clustering
    # DetectCommunity_nk(G_nk_dict)
    # # ClusteringCoefficient_nk("mean", G_nk_dict)
    # # ClusteringCoefficient_nk("global", G_nk_dict)
    # ClusteringF1Similarity_nk(G_nk_dict)
    
    # # # Application level
    # Centrality_nk("PageRank", G_nk_dict, 100)
    # # Centrality_gt("PageRank", G_gt_dict, 100)
    # # max_flow_gt(G_gt_dict)
    # # min_st_cut_gt(G_gt_dict)
    
    # # # Misc
    # QuadraticFormSimilarity_nk(G_nk_dict)
    

    # G_gt_dict = readAllGraphsGT(dataset_name, config) 
    # print(G_gt_dict)

    # ApproximateDiameter_gt(G_gt_dict)
    # SPSP_gt(G_gt_dict, num_nodes=100)
    # GlobalClusteringCoefficient_gt(G_gt_dict)
    # LocalClusteringCoefficient_gt(G_gt_dict)

    # Centrality_gt("Betweenness", G_gt_dict, 100)
    # Centrality_gt("Closeness", G_gt_dict, 100)
    # Centrality_gt("Katz", G_gt_dict, 100)
    # Centrality_gt("Eigenvector", G_gt_dict, 100)

    # NOT USED
    # Centrality("SciPyPageRank", 10)
    # Centrality("SciPyEVZ", 10)
    # Centrality("ApproxElectricalCloseness", 10)
    # min_cut(G_gt)
    # EigenValueSimilarity(G)
"""