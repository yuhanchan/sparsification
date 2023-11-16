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
# import cProfile, pstats, io
# from pstats import SortKey
import time, timeit
import sys
import psutil
import json
import sparsifier
from memory_profiler import memory_usage

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ") 
    print("please source env.sh at the top level of the project")
    exit(1)

def getOutputFile(dataset_name, config, sparsifier_name, prune_rate, postfix_folder, undirected_only=False):
    if undirected_only:
        if config[dataset_name]["weighted"]:
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/udw.wel"
        else:
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/uduw.el"
    else:
        if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/dw.wel"
        elif config[dataset_name]["directed"]: 
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/duw.el"
        elif config[dataset_name]["weighted"]:
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/udw.wel"
        else:
            output_file = f"{PROJECT_HOME}/data/{dataset_name}/pruned/{sparsifier_name}/{prune_rate}/{postfix_folder}/uduw.el"
    return output_file


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        # mem_before = process_memory()
        t_s, pt_s = time.time(), time.process_time()
        # result = func(*args, **kwargs)
        mem_usage, result = memory_usage((func, args, kwargs), retval=True, max_iterations=1, )
        t_e, pt_e = time.time(), time.process_time()
        # mem_after = process_memory()
        # print(f"{func.__name__}:consumed memory: {mem_after-mem_before:,}, wall time: {t_e-t_s} s, cpu time: {pt_e-pt_s} s")
        print(f"{func.__name__}:consumed memory: {max(mem_usage):,}, wall time: {t_e-t_s} s, cpu time: {pt_e-pt_s} s")
        return result
    return wrapper


# @profile
def KNeighborSparsifier(dataset_name, G, k, config, postfix_folder="0", print_info=False):
    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    # select K random neighbors for each node
    for node in range(0, G.numberOfNodes()):
        neighbors = list(G.iterNeighbors(node))
        if len(neighbors) > k:
            neighbors = random.sample(neighbors, k)
        for neighbor in neighbors:
            if not new_G.hasEdge(node, neighbor):
                if G.isWeighted():
                    new_G.addEdge(node, neighbor, w=G.weight(node, neighbor))
                else:
                    new_G.addEdge(node, neighbor)

    prune_rate = round(1 - new_G.numberOfEdges() / G.numberOfEdges(), 3)
    if print_info:
        print()
        nk.overview(new_G)
    print(f"k = {k}, prune_rate = {prune_rate}")
    output_file = getOutputFile(dataset_name, config, "KNeighbor", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(new_G, output_file, nk.Format.EdgeListSpaceZero)
    del new_G
    return


# @profile
def RankDegreeSparsifier(dataset_name, G, targetRatio, config, rho=0.1, postfix_folder="0", print_info=False):
    targetNum = int(G.numberOfEdges() * targetRatio)
    print(f"\ntargetNum: {targetNum}")
    seeds = set()
    seen = set()

    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    iter_count, count = 0, 0
    while new_G.numberOfEdges() < targetNum:
        iter_count += 1
        # print(f"new_G.numberOfEdges(): {new_G.numberOfEdges()}")
        if not seeds:
            # generate random seeds from all_node_set
            seeds = random.sample(list(range(0, G.numberOfNodes())), 10)
            seeds = set(seeds)
        
        if count > int(0.01*targetNum):
            iter_count = 0
            count = 0
        elif iter_count > 1000 and count <= int(0.01*targetNum) and rho < 0.99: # less than 10 edges added in the last 1000 iterations
            iter_count = 0
            rho += 0.1
            seen = set()
            count = 0
            print(f"increasing rho: {rho:.3}, new_G.numberOfEdges(): {new_G.numberOfEdges()}")

        new_seeds = set()
        for seed in seeds:
            if seed in seen:
                continue
            seen.add(seed)
            # get neighbors
            neighbors = list(G.iterNeighbors(seed))
            # rank neighbors by degree
            neighbors = sorted(neighbors, key=lambda x: G.degree(x), reverse=True)
            # select top rho * len(neighbors)
            neighbors = neighbors[:max(1, int(len(neighbors) * rho))]
            # add edges
            for neighbor in neighbors:
                if not new_G.hasEdge(seed, neighbor):
                    if G.isWeighted():
                        new_G.addEdge(seed, neighbor, w=G.weight(seed, neighbor))
                    else:
                        new_G.addEdge(seed, neighbor)
                    new_seeds.add(neighbor)
                    count += 1
        seeds = new_seeds
    if print_info:
        nk.overview(new_G)
    prune_rate = round(1 - new_G.numberOfEdges() / G.numberOfEdges(), 3)
    output_file = getOutputFile(dataset_name, config, "RankDegree", prune_rate, postfix_folder)
    print(f"\nRankDegree, prune_rate = {prune_rate}:")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(new_G, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def CutSparsifier(dataset_name, G, targetRatio, config):
    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    # find k for each edge, where k is the max connectivity of the edge
    k = {}
    for edge in G.iterEdges():
        k[edge] = 10

    epsilon = 10
    d = 1
    rho = 16*(d+2)*np.log(G.numberOfNodes()) / epsilon**2
    print(rho)

    # include all edges with probability p
    for edge in G.iterEdges():
        p = min(1, rho / k[edge])
        if random.random() < p:
            new_G.addEdge(*edge)
    return new_G


# @profile
def RandomEdgeSparsifier(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    randomEdgeSparsifier = nk.sparsification.RandomEdgeSparsifier()
    randomGraph = randomEdgeSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nRandom, prune_rate = {prune_rate}:")
    if print_info:
        nk.overview(randomGraph)
    output_file = getOutputFile(dataset_name, config, "Random", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(randomGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def LocalDegreeSparsifier(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    """
        For directed graph, based on out-degree
    """
    localDegSparsifier = nk.sparsification.LocalDegreeSparsifier()
    localDegGraph = localDegSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nLocalDegree, prune_rate = {prune_rate}:")
    if print_info: 
        nk.overview(localDegGraph)
    output_file = getOutputFile(dataset_name, config, "LocalDegree", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(localDegGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def ForestFireSparsifier(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    if targetRatio > 0.5: # high keep ratio requires low node burn rate
        fireSparsifier = nk.sparsification.ForestFireSparsifier(0.2, 5.0)
    else:
        fireSparsifier = nk.sparsification.ForestFireSparsifier(0.6, 5.0)
    fireGraph = fireSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nForestFire, prune_rate = {prune_rate}:")
    if print_info:
        nk.overview(fireGraph)
    output_file = getOutputFile(dataset_name, config, "ForestFire", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(fireGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def LocalSimilaritySparsifier(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    localSimilaritySparsifier = nk.sparsification.LocalSimilaritySparsifier()
    localSimilarityGraph = localSimilaritySparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nLocalSimilarity, prune_rate = {prune_rate}:")
    if print_info:
        nk.overview(localSimilarityGraph)
    output_file = getOutputFile(dataset_name, config, "LocalSimilarity", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(localSimilarityGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def SCANSparsifier(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    scanSparsifier = nk.sparsification.SCANSparsifier()
    scanGraph = scanSparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nSCAN, prune_rate = {prune_rate}:")
    if print_info:
        nk.overview(scanGraph)
    output_file = getOutputFile(dataset_name, config, "SCAN", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(scanGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def LSpar(dataset_name, G, config, c=0.1, postfix_folder="0", print_info=False):
    jaccardSimilaritySparsifier = nk.sparsification.JaccardSimilaritySparsifier()
    scores = jaccardSimilaritySparsifier.scores(G)
    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted(), directed=G.isDirected())
    for node in list(range(G.numberOfNodes())):
        # get the top k neighbors
        neighbors = list(G.iterNeighbors(node))
        d = len(neighbors)
        # dictionary of neighbors and their scores
        neighbor_scores = {neighbor: scores[neighbor] for neighbor in neighbors}
        # sort the dictionary by value
        sorted_neighbor_scores = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)
        # add the top d^c neighbors to the new graph
        for neighbor, score in sorted_neighbor_scores[:int(d**c)]:
            if not new_G.hasEdge(node, neighbor):
                if G.isWeighted():
                    new_G.addEdge(node, neighbor, w=G.weight(node, neighbor))
                else:
                    new_G.addEdge(node, neighbor)
    if print_info:
        print()
        nk.overview(new_G)
    prune_rate = round(1 - new_G.numberOfEdges() / G.numberOfEdges(), 3)
    output_file = getOutputFile(dataset_name, config, "LSpar", prune_rate, postfix_folder)
    print(f"c: {c:.3} , prune rate: {prune_rate}")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(new_G, output_file, nk.Format.EdgeListSpaceZero)
    del new_G
    return


# @profile
def GSpar(dataset_name, G, targetRatio, config, postfix_folder="0", print_info=False):
    jaccardSimilaritySparsifier = nk.sparsification.JaccardSimilaritySparsifier()
    jaccardGraph = jaccardSimilaritySparsifier.getSparsifiedGraphOfSize(G, targetRatio)
    prune_rate = round(1-targetRatio, 3)
    print(f"\nGSpar, prune_rate = {prune_rate}:")
    if print_info:
        nk.overview(jaccardGraph)
    output_file = getOutputFile(dataset_name, config, "GSpar", prune_rate, postfix_folder)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(jaccardGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def SpanningForestSparsifier(dataset_name, G, config, postfix_folder="0", print_info=False):
    spanningGraph = nk.graph.SpanningForest(G).run().getForest()
    output_file = getOutputFile(dataset_name, config, "SpanningForest", "", postfix_folder, undirected_only=True)
    print("\nspanning:")
    nk.overview(spanningGraph)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(spanningGraph, output_file, nk.Format.EdgeListSpaceZero)


# @profile
def GreedySpannerSparsifier(dataset_name, G, t, config, postfix_folder="0", print_info=False):
    # create a new graph with same number of nodes as originalGraph, but no edges
    new_G = nk.graph.Graph(G.numberOfNodes(), weighted=G.isWeighted())

    N = G.numberOfEdges()

    G_copy = copy.deepcopy(G)
    while G_copy.numberOfEdges():
        # if G_copy.numberOfEdges() % 1000 == 0:
            # print(f"{G_copy.numberOfEdges()}/{N}")
        edge = nk.graphtools.randomEdge(G_copy)
        G_copy.removeEdge(*edge)
        if nk.distance.BidirectionalDijkstra(G_copy, edge[0], edge[1]).run().getDistance() > t:
            new_G.addEdge(*edge)

    output_file = getOutputFile(dataset_name, config, f"Spanner-{t}", "", postfix_folder, undirected_only=True)
    print(f"\nspanner-{t}:")
    if print_info:
        nk.overview(new_G)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(new_G, output_file, nk.Format.EdgeListSpaceZero)
        

# @profile
def Spanner(dataset_name, G, k, config, postfix_folder="0", print_info=False):
    spanner_k = GreedySpanner(G, k)
    output_file = getOutputFile(dataset_name, config, f"Spanner-{k}", "", postfix_folder, undirected_only=True)
    print(f"\nspanner-{k}:")
    if print_info:
        nk.overview(spanner_k)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(spanner_k, output_file, nk.Format.EdgeListSpaceZero)


def wrapped_spanner(*args, **kwargs):
    t_s, pt_s = time.time(), time.process_time()
    mem_usage, result = memory_usage((GreedySpannerSparsifier, args, kwargs), retval=True, max_iterations=1,)
    t_e, pt_e = time.time(), time.process_time()
    print(f"spanner: dataset: {args[0]}, k: {args[2]}, consumed memory: {max(mem_usage):,}, wall time: {t_e-t_s} s, cpu time: {pt_e-pt_s} s")
    return result


# @profile
def ERMinSparsifier(dataset_name, config, multi_process=False, postfix_folder="0"):
    if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.sym.wel"
    elif config[dataset_name]["directed"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.sym.el"
    elif config[dataset_name]["weighted"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.wel" # always use directed graph
    else:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el" # always use directed graph

    # if er files are not pre-computed, compute first
    if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/raw/stage3.npz"):
        epsilon, val = list(config[dataset_name]["ermin_epsilon_to_prune_rate_map"].items())[0]
        sparsifier.python_er_sparsify(
            osp.join(
                osp.dirname(osp.realpath(__file__)),
                file_path,
            ),
            dataset_name=dataset_name,
            dataset_type="el",
            epsilon=float(epsilon),
            prune_rate_val=val,
            reuse=True,
            method="min",
            config=config,
            postfix_folder=postfix_folder,
        )
        print("done computing effective resistance, now rerun to do sampling for all epsilon in config file")
        return

    if multi_process:
        with ProcessPoolExecutor(max_workers=128) as executor:
            futures = {}
            for epsilon, val in config[dataset_name][
                "ermin_epsilon_to_prune_rate_map"
            ].items():
                futures[
                    executor.submit(
                        sparsifier.python_er_sparsify,
                        osp.join(
                            osp.dirname(osp.realpath(__file__)),
                            file_path,
                        ),
                        dataset_name=dataset_name,
                        dataset_type="el",
                        epsilon=float(epsilon),
                        prune_rate_val=val,
                        reuse=True,
                        method="min",
                        config=config,
                        postfix_folder=postfix_folder,
                    )
                ] = epsilon

            for future in futures:
                print(f"start {futures[future]}")
                try:
                    future.result()
                except Exception as e:
                    print(e)
                    print(f"failed {futures[future]}")
                    sys.exit(1)
    else:
        for epsilon, val in config[dataset_name][
            "ermin_epsilon_to_prune_rate_map"
        ].items():
            sparsifier.python_er_sparsify(
                osp.join(
                        osp.dirname(osp.realpath(__file__)),
                        file_path,
                    ),
                    dataset_name=dataset_name,
                    dataset_type="el",
                    epsilon=float(epsilon),
                    prune_rate_val=val,
                    reuse=True,
                    method="min",
                    config=config,
                    postfix_folder=postfix_folder,
            )


# @profile
def ERMaxSparsifier(dataset_name, config, multi_process=False, postfix_folder="0"):
    if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.sym.wel"
    elif config[dataset_name]["directed"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.sym.el"
        # file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el"
    elif config[dataset_name]["weighted"]:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.wel" # always use directed graph
    else:
        file_path = f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el" # always use directed graph

    # if er files are not pre-computed, compute first
    if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/raw/stage3.npz"):
        epsilon, val = list(config[dataset_name]["ermax_epsilon_to_prune_rate_map"].items())[0]
        sparsifier.python_er_sparsify(
            osp.join(
                osp.dirname(osp.realpath(__file__)),
                file_path,
            ),
            dataset_name=dataset_name,
            dataset_type="el",
            epsilon=float(epsilon),
            prune_rate_val=val,
            reuse=True,
            method="max",
            config=config,
            postfix_folder=postfix_folder,
        )
        print("done computing effective resistance, now rerun to do sampling for all epsilon in config file")
        return
    

    if multi_process:
        with ProcessPoolExecutor(max_workers=128) as executor:
            futures = {}
            for epsilon, val in config[dataset_name][
                "ermax_epsilon_to_prune_rate_map"
            ].items():
                futures[
                    executor.submit(
                        sparsifier.python_er_sparsify,
                        osp.join(
                            osp.dirname(osp.realpath(__file__)),
                            file_path,
                        ),
                        dataset_name=dataset_name,
                        dataset_type="el",
                        epsilon=float(epsilon),
                        prune_rate_val=val,
                        reuse=True,
                        method="max",
                        config=config,
                        postfix_folder=postfix_folder,
                    )
                ] = epsilon

            for future in futures:
                print(f"start {futures[future]}")
                try:
                    future.result()
                except Exception as e:
                    print(e)
                    print(f"failed {futures[future]}")
                    sys.exit(1)
    else:    
        for epsilon, val in config[dataset_name][
            "ermax_epsilon_to_prune_rate_map"
            ].items():
            sparsifier.python_er_sparsify(
                osp.join(
                    osp.dirname(osp.realpath(__file__)),
                    file_path,
                    ),
                    dataset_name=dataset_name,
                    dataset_type="el",
                    epsilon=float(epsilon),
                    prune_rate_val=val,
                    reuse=True,
                    method="max",
                    config=config,
                    postfix_folder=postfix_folder,
            )



