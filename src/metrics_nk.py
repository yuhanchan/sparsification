import networkit as nk
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
from memory_profiler import memory_usage
import errno
import signal
import functools


class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                t_s = time.time()
                result = func(*args, **kwargs)
                t_e = time.time()
                print(f"{func.__name__} finished after time: {t_e-t_s:.2f} s")
            except:
                print(f"{func.__name__} timeout after {seconds} seconds")
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ") 
    print("please source env.sh at the top level of the project")
    exit(1)

MAX_TIMEOUT = 86400 # timeout after 1 day

def compute_transition_matrix(adj_matrix):
    out_degrees = np.sum(adj_matrix, axis=1)
    out_degrees[out_degrees == 0] = 1  # Avoid division by zero
    return (adj_matrix / out_degrees[:, np.newaxis]).T


def add_damping(transition_matrix, damping_factor=0.85):
    n = transition_matrix.shape[0]
    return (damping_factor * transition_matrix) + ((1 - damping_factor) / n)


def power_iteration(transition_matrix, max_iter=100, tol=1e-6):
    n = transition_matrix.shape[0]
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_next = np.dot(transition_matrix, pr)
        if np.linalg.norm(pr_next - pr) < tol:
            break
        pr = pr_next
    return pr


def page_rank(G, damping_factor=0.85, max_iter=100, tol=1e-6):
    adj_matrix = np.array(nk.algebraic.adjacencyMatrix(G).todense())
    transition_matrix = compute_transition_matrix(adj_matrix)
    damped_matrix = add_damping(transition_matrix, damping_factor)
    pr = power_iteration(damped_matrix, max_iter, tol)
    return pr


def cos_sim(a, b):
    return np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def ranking_precision(a, b, k=100):
    a_rank = np.argsort(a)[::-1]
    b_rank = np.argsort(b)[::-1]
    return len(set(a_rank[:k]).intersection(set(b_rank[:k]))) / k


@timeout(MAX_TIMEOUT)
def degreeDistribution_nk(dataset_name, G_dict, nbin=20, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, "degreeDistribution", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")


    degrees = nk.centrality.DegreeCentrality(G_dict["original"][0], normalized=True, outDeg=True, ignoreSelfLoops=True).run().scores()
    histogram = np.histogram(degrees, bins=nbin)
    baseline_dist = histogram[0] / sum(histogram[0])

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            degrees = nk.centrality.DegreeCentrality(Graph, normalized=True, outDeg=True, ignoreSelfLoops=True).run().scores()
            histogram = np.histogram(degrees, bins=nbin)
            degree_dist = histogram[0] / sum(histogram[0])

            # compute Bhattacharyya distance
            Bhattacharyya_distance = -np.log(np.sum(np.sqrt(degree_dist * baseline_dist)))
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\t Bhattacharyya_distance: {Bhattacharyya_distance:.3f}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\t Bhattacharyya_distance: {Bhattacharyya_distance:.3f}")

    if logToFile:
        fout.close()
    

@timeout(MAX_TIMEOUT)
def EffectiveDiameter_nk(G_dict, approximate=False): # only for undirected graphs
    print("90 percent Effective Diameter, set approximate=True to use approximate algorithm (faster)")
    print("Warn: Only showing the diameter of the largest connected component")
    ranks = []
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            if Graph.isDirected():
                cc = nk.components.WeaklyConnectedComponents(Graph)
                cc.run()
                # get the largest connected component id
                largestCC_id = sorted(cc.getComponentSizes().items(), key=lambda x: x[1], reverse=True)[0][0]
                largestCC_node_list = cc.getComponents()[largestCC_id]
                # get subgraph with largestCC_node_list
                largestCC = nk.graphtools.subgraphFromNodes(Graph, largestCC_node_list)
            else:
                cc = nk.components.ConnectedComponents(Graph)
                cc.run()
                largestCC = cc.extractLargestConnectedComponent(Graph, True)

            if approximate:
                diameter = nk.distance.EffectiveDiameterApproximation(largestCC, ratio=0.9).run().getEffectiveDiameter()
            else:
                diameter = nk.distance.EffectiveDiameter(largestCC, ratio=0.9).run().getEffectiveDiameter()
            ranks.append((name, diameter))
            print(f"{name}\t#nodes: {largestCC.numberOfNodes()} ({largestCC.numberOfNodes()*100.0/Graph.numberOfNodes():.2f}%)\t #edges: {largestCC.numberOfEdges()} ({largestCC.numberOfEdges()*100.0/Graph.numberOfEdges():.2f}%)\t diameter: {diameter}")



@timeout(MAX_TIMEOUT)
def Eccentricity_nk(G_dict, num_nodes=100): # only for undiredted graphs

    # pick random nodes
    num_nodes = num_nodes if num_nodes < G_dict["original"][0].numberOfNodes() else G_dict["original"][0].numberOfNodes()
    srcs = random.sample(list(range(0, G_dict["original"][0].numberOfNodes())), num_nodes)
    baseline = [nk.distance.Eccentricity().getValue(G_dict["original"][0], src)[1] for src in srcs]

    baseline_dist = np.zeros(max(baseline)+1)
    ecc, numberOfNodes = np.unique(baseline, return_counts=True)
    for i, e in enumerate(ecc):
        baseline_dist[int(e)] = numberOfNodes[i] / num_nodes
    baseline_dist_grouped = []
    for i in range(20):
        baseline_dist_grouped.append(np.sum(baseline_dist[int(i*len(baseline_dist)/20):int((i+1)*len(baseline_dist)/20)]))
    baseline_dist_grouped = np.array(baseline_dist_grouped)
    
    ranks = []
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            eccentricity = [nk.distance.Eccentricity().getValue(Graph, src)[1] for src in srcs]

            eccentricity_dist = np.zeros(max(eccentricity)+1)
            ecc, numberOfNodes = np.unique(eccentricity, return_counts=True)
            for i, e in enumerate(ecc):
                eccentricity_dist[int(e)] = numberOfNodes[i] / num_nodes
            eccentricity_dist_grouped = []
            for i in range(20):
                eccentricity_dist_grouped.append(np.sum(eccentricity_dist[int(i*len(eccentricity_dist)/20):int((i+1)*len(eccentricity_dist)/20)]))
            eccentricity_dist_grouped = np.array(eccentricity_dist_grouped)

            # compute Bhattacharyya distance
            Bhattacharyya_distance = -np.log(np.sum(np.sqrt(baseline_dist_grouped * eccentricity_dist_grouped)))

            ratio = list(map(lambda x: x[0]/x[1] if x[1]!=0 else np.nan, zip(eccentricity, baseline)))
            ratio_mean = np.ma.masked_invalid(ratio).mean()
            ratio_min = np.ma.masked_invalid(ratio).min()
            ratio_max = np.ma.masked_invalid(ratio).max()
            ratio_std = np.ma.masked_invalid(ratio).std()
            ranks.append((name, ratio_mean))
        
            print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()}\t ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t Bhattacharyya distance: {Bhattacharyya_distance:.3f}")



@timeout(MAX_TIMEOUT)
def SPSP_nk(G_dict, num_nodes=100): # only for undirected graphs
    # generate num_nodes different random int 
    srcs = random.sample(list(range(0, G_dict["original"][0].numberOfNodes())), num_nodes)
    baseline = nk.distance.SPSP(G_dict["original"][0], srcs).run().getDistances()
    baseline = list(chain.from_iterable(baseline))
    baseline = list(map(lambda x: x if x < 1e10 else float('inf'), baseline)) 
    print("total number of pairs:", len(baseline), end="")
    print(f", unreachable ratio: {baseline.count(float('inf'))/len(baseline):.3f}")

    ranks = []
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            distances = nk.distance.SPSP(Graph, srcs).run().getDistances()
            distances = list(chain.from_iterable(distances))
            distances = list(map(lambda x: x if x < 1e10 else float('inf'), distances)) 
            ratio = list(map(lambda x: x[0]/x[1] if x[1]!=0 and x[1] != float('inf') else 1, zip(distances, baseline)))
            # tmp = [(baseline[i], distances[i]) for i, x in enumerate(ratio) if x == 0]
            # print(tmp, len(tmp))
            ratio_mean = np.ma.masked_invalid(ratio).mean()
            ratio_min = np.ma.masked_invalid(ratio).min()
            ratio_max = np.ma.masked_invalid(ratio).max()
            ratio_std = np.ma.masked_invalid(ratio).std()
            ranks.append((name, ratio_mean))
            print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()},\t ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t unreachable ratio: {distances.count(float('inf'))/len(ratio):.3f}")


@timeout(MAX_TIMEOUT)
def Centrality_nk(dataset_name, algo_, G_dict, topN, logToFile=False):
    t_s, pt_s = time.time(), time.process_time()

    pos_params = ()
    dict_params = {}
    if algo_ == "Betweenness":
        algo = nk.centrality.Betweenness
    elif algo_ == "ApproxBetweenness":
        algo = nk.centrality.ApproxBetweenness 
    elif algo_ == "EstimateBetweenness":
        algo = nk.centrality.EstimateBetweenness
        pos_params = (500,) # nsample
    elif algo_ == "DynBetweenness":
        algo = nk.centrality.DynBetweenness
    elif algo_ == "DynApproxBetweenness":
        algo = nk.centrality.DynApproxBetweenness 
    elif algo_ == "Closeness":
        algo = nk.centrality.Closeness
        pos_params = (True, True) # normalized, checkConnectedness
    elif algo_ == "ApproxCloseness":
        algo = nk.centrality.ApproxCloseness 
        pos_params = (True, True) # normalized, checkConnectedness
    elif algo_ == "TopCloseness":
        algo = nk.centrality.TopCloseness 
        dict_params = {"k": 100}
    elif algo_ == "Degree":
        algo = nk.centrality.DegreeCentrality
    elif algo_ == "KPath":
        algo = nk.centrality.KPathCentrality
    elif algo_ == "Katz":
        algo = nk.centrality.KatzCentrality
        dict_params = {"alpha": 0.5}
    elif algo_ == "LocalClusteringCoefficient":
        algo = nk.centrality.LocalClusteringCoefficient
    elif algo_ == "Laplacian":
        algo = nk.centrality.LaplacianCentrality
    elif algo_ == "PageRank":
        algo = nk.centrality.PageRank
    elif algo_ == "Eigenvector":
        algo = nk.centrality.EigenvectorCentrality
    elif algo_ == "ApproxElectricalCloseness":
        algo = nk.centrality.ApproxElectricalCloseness
    elif algo_ == "CoreDecomposition":
        algo = nk.centrality.CoreDecomposition
    elif algo_ == "SciPyEVZ":
        algo = nk.centrality.SciPyEVZ
    elif algo_ == "SciPyPageRank":
        algo = nk.centrality.SciPyPageRank
    else:
        raise NotImplementedError
    
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"{algo_}Centrality", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    print(f"{algo_} Centrality")
    
    if algo_ == "CoreDecomposition":
        G_dict["original"][0].removeSelfLoops()
    if algo_ == "TopCloseness":
        baseline = algo(G_dict["original"][0], *pos_params, **dict_params).run().topkNodesList()
    else:
        baseline = algo(G_dict["original"][0], *pos_params, **dict_params).run().ranking()
        baseline = [x[0] for x in baseline]

    ranks = []
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            if algo_ == "CoreDecomposition":
                Graph.removeSelfLoops()

            if algo_ == "TopCloseness":
                res = algo(Graph, *pos_params, **dict_params).run().topkNodesList()
            else:
                res = algo(Graph, *pos_params, **dict_params).run().ranking()
                res = [x[0] for x in res]

            intersect = set(baseline[:topN]).intersection(set(res[:topN]))
            baseline_ind = [baseline.index(x) for x in intersect]
            res_ind = [res.index(x) for x in intersect]

            topN_precision = len(intersect) / topN
            topN_correlation = scipy.stats.spearmanr(baseline_ind, res_ind)[0]
            
            ranks.append((name, topN_precision))
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()}\t top_{topN}_precision: {topN_precision:.2f}\t top_{topN}_correlation: {topN_correlation:.2f}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()}\t top_{topN}_precision: {topN_precision:.2f}\t top_{topN}_correlation: {topN_correlation:.2f}")

    t_e, pt_e = time.time(), time.process_time()
    print(f"Time: {t_e-t_s:.2f} s\t Process Time: {pt_e-pt_s:.2f} s")

    if logToFile:
        fout.close()
    

@timeout(MAX_TIMEOUT)
def ClusteringCoefficient_nk(algo_, G_dict): # only for undirected graphs
    if algo_ == "mean": # mean of local clustering coefficient
        ranks = []
        for name, Graphs in G_dict.items():
            for Graph in Graphs:
                Graph.removeSelfLoops()
                res = np.mean(nk.centrality.LocalClusteringCoefficient(Graph).run().scores())
                ranks.append((name, res))
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()},\t mean Clustering Coefficient: {res:.3f}")
    elif algo_ == "global":
        ranks = []
        for name, Graphs in G_dict.items():
            for Graph in Graphs:
                res = nk.globals.ClusteringCoefficient(Graph).exactGlobal(Graph)
                ranks.append((name, res))
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t #edges: {Graph.numberOfEdges()},\t global Clustering Coefficient: {res:.3f}")
    print()


@timeout(MAX_TIMEOUT)
def ClusteringF1Similarity_nk(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, "ClusteringF1Similarity", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    reference_C = nk.community.LFM(G_dict["original"][0], nk.scd.LFMLocal(G_dict["original"][0])).run().getCover()

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            C = nk.community.LFM(Graph, nk.scd.LFMLocal(Graph)).run().getCover()
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\tF1_Similarity: {nk.community.CoverF1Similarity(Graph, C, reference_C).run().getUnweightedAverage()}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\tF1_Similarity: {nk.community.CoverF1Similarity(Graph, C, reference_C).run().getUnweightedAverage()}")

    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def ClusteringF1SimilarityWithGroundTruth_nk(dataset_name, G_dict, groundTruthFile, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, "ClusteringF1SimilarityWithGroundTruth", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    # reference_C = nk.community.LFM(G_dict["original"][0], nk.scd.LFMLocal(G_dict["original"][0])).run().getCover()
    reference_C = nk.Cover(n=G_dict["original"][0].numberOfNodes())
    reference_C.setUpperBound(G_dict["original"][0].numberOfNodes())
    with open(groundTruthFile, "r") as f:
        s = 0
        for line in f:
            line = line.strip().split()
            if line:
                for node in line:
                    reference_C.addToSubset(s, int(node))
                s += 1

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            C = nk.community.LFM(Graph, nk.scd.LFMLocal(Graph)).run().getCover()
            # C = nk.community.CutClustering(Graph).run().getCover()
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\tF1_Similarity: {nk.community.CoverF1Similarity(Graph, C, reference_C).run().getUnweightedAverage()}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\tF1_Similarity: {nk.community.CoverF1Similarity(Graph, C, reference_C).run().getUnweightedAverage()}")
        
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def DetectCommunity_nk(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"DetectCommunity", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    for name, Graphs in G_dict.items():
        print(name)
        for Graph in Graphs:
            C = nk.community.detectCommunities(Graph)
            numCommunities = C.numberOfSubsets()
            modularity = nk.community.Modularity().getQuality(C, Graph)
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\t #communities: {numCommunities}\t modularity: {modularity:.3f}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges: {Graph.numberOfEdges()}\t #communities: {numCommunities}\t modularity: {modularity:.3f}")
                print()
    
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def QuadraticFormSimilarity_nk(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"QuadraticFormSimilarity", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    x = np.random.normal(size=(G_dict["original"][0].numberOfNodes(), 100))

    L_G = nk.algebraic.laplacianMatrix(G_dict["original"][0])
    baseline = np.diag(np.transpose(x)@L_G@x)

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            L_H = nk.algebraic.laplacianMatrix(Graph)
            res = np.diag(np.transpose(x)@L_H@x)
            
            ratio = res/baseline
            ratio_min = np.min(ratio)
            ratio_max = np.max(ratio)
            ratio_mean = np.mean(ratio)
            ratio_std = np.std(ratio)

            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges {Graph.numberOfEdges()}\tratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}\n")
            else:
                print(f"{name}\t#nodes: {Graph.numberOfNodes()}\t#edges {Graph.numberOfEdges()}\tratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}")
