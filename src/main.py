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
from graph_reader import *
from memory_profiler import memory_usage
import argparse
from functools import wraps
from multiprocessing.pool import Pool
import rich

 
PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def graphOverview(G):
    """Print graph overview, helper function """
    for name, Graph in G.items():
        nk.overview(Graph)

def loadOriginalGraph(dataset_name, config, undirected_only=False):
    """Load original graph from file

    Args:
        dataset_name (str): dataset name
        config (dict): config loaded from json
        undirected_only (bool, optional): Set to True to override graph directness in config file and load undirected graph only. 
                                          Defaults to False. This is used for sparsifiers that only support undirected graph.

    Returns:
        nk graph: original graph
    """
    if dataset_name not in config:
        raise ValueError(f"dataset {dataset_name} not in config, check config.json")

    if undirected_only:
        if config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
        else:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)
    else:
        if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/dw.wel", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["directed"]:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/duw.el", nk.Format.EdgeListSpaceZero, directed=True)
        elif config[dataset_name]["weighted"]:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/udw.wel", nk.Format.EdgeListSpaceZero, directed=False)
        else:
            originalGraph = nk.readGraph(f"{PROJECT_HOME}/data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero, directed=False)

    nk.overview(originalGraph)
    nk.graph.Graph.indexEdges(originalGraph)
    return originalGraph


def graphSparsifier(dataset_names, sparsifiers, targetRatios, config, multi_process=False, max_workers=8, num_run=1):
    """This function generates sparsified graphs with different sparsification algorithms, and save to files

    Args:
        dataset_name (str or list): dataset name
        sparsifiers (list or str): list of sparsifiers to run
        targetRatios (list or float): target ratios
        config (dict): config loaded from json
        multi_process (bool, optional): Set to True to run sparsifiers in parallel. Defaults to False.
        max_workers (int, optional): Number of workers for parallelization, only effective when multi_process set to True. Defaults to 8.
        num_run (int, optional): Number of runs for non-deterministic sparsifiers, deterministic sparsifiers will ignore this. Defaults to 1.
    """

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    if isinstance(sparsifiers, str):
        sparsifiers = [sparsifiers]
    if isinstance(targetRatios, float):
        sparseRatios = [targetRatios]

    original_stdout = sys.stdout
    for dataset_name in dataset_names:
        # redirect output to log file
        outfile_path = f"{PROJECT_HOME}/output_sparsifier_raw/{dataset_name}.txt"
        os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
        sys.stdout = open(outfile_path, "w")
        originalGraph = loadOriginalGraph(dataset_name, config, undirected_only=False)
        originalGraph_ud = loadOriginalGraph(dataset_name, config, undirected_only=True)
        for sparsifier in sparsifiers:
            if sparsifier == "ER":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running ER sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/raw/stage3.npz"):
                    ERSparsifier(dataset_name, config, multi_process=False, postfix_folder=0)
                    for i in range(num_run):
                        ERSparsifier(dataset_name, config, multi_process=True, postfix_folder=str(i))
                else:
                    for i in range(num_run):
                        ERSparsifier(dataset_name, config, multi_process=True, postfix_folder=str(i))

            elif sparsifier == "Random":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running Random sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            for i in range(num_run):
                                executor.submit(RandomEdgeSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        for i in range(num_run):
                            RandomEdgeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))

            elif sparsifier == "ForestFire":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running ForestFire sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            for i in range(num_run):
                                executor.submit(ForestFireSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        for i in range(num_run):
                            ForestFireSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))

            elif sparsifier == "RankDegree":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running RankDegree sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            for i in range(num_run):
                                executor.submit(RankDegreeSparsifier, dataset_name, originalGraph, targetRatio, config, str(i))
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        for i in range(num_run):
                            RankDegreeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder=str(i))

            elif sparsifier == "LocalDegree":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running LocalDegree sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            executor.submit(LocalDegreeSparsifier, dataset_name, originalGraph, targetRatio, config, "0")
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        LocalDegreeSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")

            elif sparsifier == "GSpar":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running GSpar sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            executor.submit(GSpar, dataset_name, originalGraph, targetRatio, config, "0")
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        GSpar(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")

            elif sparsifier == "LocalSimilarity":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running Local Similarity sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            executor.submit(LocalSimilaritySparsifier, dataset_name, originalGraph, targetRatio, config, "0")
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        LocalSimilaritySparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")

            elif sparsifier == "SCAN":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running SCAN sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for targetRatio in targetRatios:
                            targetRatio = round(targetRatio, 3)
                            executor.submit(SCANSparsifier, dataset_name, originalGraph, targetRatio, config, "0")
                else:
                    for targetRatio in targetRatios:
                        targetRatio = round(targetRatio, 3)
                        SCANSparsifier(dataset_name, originalGraph, targetRatio, config, postfix_folder="0")

            elif sparsifier == "KNeighbor":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running KNeighbor sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    for k in config[dataset_name]["kNeighbors"]:
                        for i in range(num_run):
                            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                                executor.submit(KNeighborSparsifier, dataset_name, originalGraph, k, config, str(i))
                else:
                    for k in config[dataset_name]["kNeighbors"]:
                        for i in range(num_run):
                            KNeighborSparsifier(dataset_name, originalGraph, k, config, postfix_folder=str(i))

            elif sparsifier == "tSpanner":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running t-spanner sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for t in [3, 5, 7]:
                        executor.submit(executor.submit(wrapped_spanner, dataset_name, originalGraph_ud, t, config, "0"))

            elif sparsifier == "SpanningForest":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running Spanning Forest sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                SpanningForestSparsifier(dataset_name, originalGraph_ud, config, postfix_folder="0")

            elif sparsifier == "LSpar":
                sys.stdout, original_stdout = original_stdout, sys.stdout
                rich.print("[bold green]Running LSpar sparsifier...[/bold green]")
                sys.stdout, original_stdout = original_stdout, sys.stdout
                if multi_process:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for c in config[dataset_name]["LSpar_c"]:
                            executor.submit(LSpar, dataset_name, originalGraph, config, c, "0")
                else:
                    for c in config[dataset_name]["LSpar_c"]:
                        LSpar(dataset_name, originalGraph, config, c, postfix_folder="0")
            
            else:
                raise ValueError("Unknown sparsifier: {}".format(sparsifier))
        
        # reset stdout
        sys.stdout = original_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="name of the dataset, or 'all'")
    parser.add_argument("--mode", choices=["sparsify", "eval", "all", "clean"], required=True, help="sparsify: only sparsify graphs; eval: only evaluate metrics (need to have run sparsify first); all: run both sparsify and eval; clean: clean all data for given [dataset_name]")
    args = parser.parse_args()

    config = json.load(open(f"{PROJECT_HOME}/config.json", "r"))

    if args.dataset_name == "all":
        dataset_names = ["ego-Facebook", 
                        "ego-Twitter", 
                        "human_gene2", 
                        "com-DBLP", 
                        "com-Amazon", 
                        "email-Enron", 
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

    if args.mode == "clean":
        print(f"About to delete raw graph, sparsified graph, sparsifier results and evaluation results for {dataset_names}. Are you sure? (y/n)")
        if input() == "y":
            for dataset_name in dataset_names:
                os.system(f"rm -rf {PROJECT_HOME}/data/{dataset_name}")
                os.system(f"rm -rf {PROJECT_HOME}/output_metric_raw/{dataset_name}")
                os.system(f"rm -rf {PROJECT_HOME}/output_metric_parsed/{dataset_name}")
                os.system(f"rm -rf {PROJECT_HOME}/output_sparsifier_raw/{dataset_name}.txt")
                os.system(f"rm -rf {PROJECT_HOME}/output_sparsifier_parsed/{dataset_name}.csv")
            print(f"Deleted {dataset_names}.")
        else:
            print("Aborted.")
        exit(0)



    if args.mode == "sparsify" or args.mode == "all":
        # for dataset_name in dataset_names:
        rich.print("[bold green]Running sparsifier...[/bold green]")
        graphSparsifier(dataset_names=dataset_names, 
                        sparsifiers=["ER", "Random", "ForestFire", "RankDegree", 
                                        "LocalDegree", "GSpar", "LocalSimilarity", 
                                        "SCAN", "KNeighbor", "tSpanner", 
                                        "SpanningForest", "LSpar"], 
                        targetRatios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                        config=config,
                        multi_process=False,
                        max_workers=3,
                        num_run=1
                        )

        rich.print("[bold green]\nRunning sparsifier done...[/bold green]")
        rich.print("[bold green]\nConverting pruned graph...[/bold green]")
        for dataset_name in dataset_names:
            os.system(f"python {PROJECT_HOME}/utils/convert_edgelist.py --dataset_name {dataset_name} --num_thread 8")
        rich.print("[bold green]Converting pruned graph done...[/bold green]")

    if args.mode == "eval" or args.mode == "all":
        for dataset_name in dataset_names:
            G_nk_dict = readAllGraphsNK(dataset_name, config)
            # check how much memory is used by this program
            # print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB")
            # exit(0)
            try:
                rich.print("[bold green]Evaluating Degree Distribution...[/bold green]")
                degreeDistribution_nk(dataset_name, G_nk_dict, nbin=100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Betweenness Centrality...[/bold green]")
                Centrality_nk(dataset_name, "EstimateBetweenness", G_nk_dict, 100, logToFile=True)
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Closeness Centrality...[/bold green]")
                Centrality_nk(dataset_name, "TopCloseness", G_nk_dict, 100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Katz Centrality...[/bold green]")
                Centrality_nk(dataset_name, "Katz", G_nk_dict, 100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Eigenvector Centrality...[/bold green]")
                Centrality_nk(dataset_name, "Eigenvector", G_nk_dict, 100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Community...[/bold green]")
                DetectCommunity_nk(dataset_name, G_nk_dict, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Clustering F1 Similarity...[/bold green]")
                ClusteringF1Similarity_nk(dataset_name, G_nk_dict, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Page Rank...[/bold green]")
                Centrality_nk(dataset_name, "PageRank", G_nk_dict, 100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Quadratic Form Similarity...[/bold green]")
                QuadraticFormSimilarity_nk(dataset_name, G_nk_dict, logToFile=True) 
            except:
                pass
            del G_nk_dict # release memory

            G_gt_dict = readAllGraphsGT(dataset_name, config) 
            try:
                rich.print("[bold green]Evaluating SPSP and Eccentricity stretch factor...[/bold green]")
                SPSP_Eccentricity_gt(dataset_name, G_gt_dict, num_nodes=100, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Global Clustering Coefficient...[/bold green]")
                GlobalClusteringCoefficient_gt(dataset_name, G_gt_dict, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Local Clustering Coefficient...[/bold green]")
                LocalClusteringCoefficient_gt(dataset_name, G_gt_dict, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Min Cut / Max Flow...[/bold green]")
                MaxFlow_gt(dataset_name, G_gt_dict, logToFile=True) 
            except:
                pass
            try:
                rich.print("[bold green]Evaluating Diameter...[/bold green]")
                ApproximateDiameter_gt(dataset_name, G_gt_dict, logToFile=True) 
            except:
                pass
            del G_gt_dict # release memory


if __name__ == "__main__":
    set_start_method("spawn") # must write this line under if __name__ == "__main__":, otherwise it will not work
    main() 
