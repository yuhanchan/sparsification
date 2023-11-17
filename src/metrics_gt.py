import graph_tool.all as gt
import graph_tool
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

@timeout(MAX_TIMEOUT)
def ApproximateDiameter_gt(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"ApproximateDiameter", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            # diameter = graph_tool.topology.pseudo_diameter(Graph, source=None, weights=Graph.edge_properties["weight"])
            diameters = []
            for i in range(10):
                s = random.randint(0, Graph.num_vertices())
                # print(s, Graph.vertex(s).out_degree())
                # while Graph.vertex(s).out_degree() == 0:
                    # s = random.randint(0, Graph.num_vertices())
                diameter = graph_tool.topology.pseudo_diameter(Graph, source=s, weights=Graph.edge_properties["weight"])
                diameters.append(diameter[0])

            diameter = np.mean(diameters)
            # print(diameters, diameter)

            if logToFile:
                fout.write(f"{name}\t {Graph}\t diameter: {diameter}\n")
            else:
                print(f"{name}\t{Graph}\tdiameter: {diameter}")
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def SPSP_Eccentricity_gt(dataset_name, G_dict, num_nodes=100, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"SPSP_Eccentricity", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    srcs = random.sample(list(range(0, G_dict["original"][0].num_vertices())), num_nodes)
    trgs = random.sample(list(range(0, G_dict["original"][0].num_vertices())), 1000)
    trgs = [t for t in trgs if t not in srcs]

    distance_baseline = np.array([])
    eccentricity_baseline = np.array([])
    for src in srcs:
        d = graph_tool.topology.shortest_distance(G_dict["original"][0], source=src, target=trgs, weights=G_dict["original"][0].edge_properties["weight"])
        distance_baseline = np.append(distance_baseline, d)
        masked_d = [x for x in d if x != float("inf")]
        if masked_d and max(masked_d) != 0:
            eccentricity_baseline = np.append(eccentricity_baseline, max(masked_d))
        else:
            eccentricity_baseline = np.append(eccentricity_baseline, float("inf"))


    # eccentricity_baseline_dist = np.zeros(int(max([e for e in eccentricity_baseline if e != float("inf")])+1))
    # ecc, numberOfNodes = np.unique(eccentricity_baseline, return_counts=True)
    # for i, e in enumerate(ecc):
    #     eccentricity_baseline_dist[int(e)] += numberOfNodes[i] / num_nodes
    # eccentricity_baseline_dist_grouped = []
    # for i in range(20):
    #     eccentricity_baseline_dist_grouped.append(np.sum(eccentricity_baseline_dist[int(i*len(eccentricity_baseline_dist)/20):int((i+1)*len(eccentricity_baseline_dist)/20)]))
    # eccentricity_baseline_dist_grouped = np.array(eccentricity_baseline_dist_grouped)

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            distances = np.array([])
            eccentricity = np.array([])
            for src in srcs:
                d = graph_tool.topology.shortest_distance(Graph, source=src, target=trgs, weights=Graph.edge_properties["weight"])
                distances = np.append(distances, d)
                masked_d = [x for x in d if x != float("inf")]
                if masked_d and max(masked_d) != 0:
                    eccentricity = np.append(eccentricity, max(masked_d))
                else:
                    eccentricity = np.append(eccentricity, float("inf"))

            # print(eccentricity)
            # eccentricity_dist = np.zeros(int(max([e for e in eccentricity if e!=float("inf")]))+1)
            # ecc, numberOfNodes = np.unique(eccentricity, return_counts=True)
            # for i, e in enumerate(ecc):
            #     eccentricity_dist[int(e)] += numberOfNodes[i] / num_nodes
            # eccentricity_dist_grouped = []
            # for i in range(20):
            #     eccentricity_dist_grouped.append(np.sum(eccentricity_dist[int(i*len(eccentricity_dist)/20):int((i+1)*len(eccentricity_dist)/20)]))
            # eccentricity_dist_grouped = np.array(eccentricity_dist_grouped)
                
            ratio = list(map(lambda x: x[0]/x[1] if x[1]!=0 and x[1] != float('inf') else float("inf"), zip(distances, distance_baseline)))
            ratio_mean = np.ma.masked_invalid(ratio).mean()
            ratio_min = np.ma.masked_invalid(ratio).min()
            ratio_max = np.ma.masked_invalid(ratio).max()
            ratio_std = np.ma.masked_invalid(ratio).std()
            if logToFile:
                fout.write(f"{name}\t{Graph}\tDistance ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t unreachable ratio: {ratio.count(float('inf'))/len(ratio):.3f}\n")
            else:
                print(f"{name}\t{Graph}\tDistance ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t unreachable ratio: {ratio.count(float('inf'))/len(ratio):.3f}")

            if [x for x in eccentricity if x != float("inf")]: # not all eccentricity are inf
                ratio = list(map(lambda x: x[0]/x[1] if x[1]!=0 and x[1] != float('inf') else float("inf"), zip(eccentricity, eccentricity_baseline)))
                ratio_mean = np.ma.masked_invalid(ratio).mean()
                ratio_min = np.ma.masked_invalid(ratio).min()
                ratio_max = np.ma.masked_invalid(ratio).max()
                ratio_std = np.ma.masked_invalid(ratio).std()
            else:
                ratio = [float("inf")]
                ratio_mean = float("inf")
                ratio_min = float("inf")
                ratio_max = float("inf")
                ratio_std = float("inf")

            if logToFile:       
                fout.write(f"{name}\t{Graph}\tEccentricity ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t isolated ratio: {ratio.count(float('inf'))/len(ratio):.3f}\n")
            else:
                print(f"{name}\t{Graph}\tEccentricity ratio min / max / mean / std: {ratio_min:.2f} / {ratio_max:.2f} / {ratio_mean:.2f} / {ratio_std:.2f}\t isolated ratio: {ratio.count(float('inf'))/len(ratio):.3f}")
            
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def Centrality_gt(algo_, G_dict, topN):
    print(f"{algo_} Centrality")
    t_s, pt_s = time.time(), time.process_time()

    if algo_ == "Betweenness":
        algo = graph_tool.centrality.betweenness
    elif algo_ == "Closeness":
        algo = graph_tool.centrality.closeness
    elif algo_ == "Katz":
        algo = graph_tool.centrality.katz
    elif algo_ == "PageRank":
        algo = graph_tool.centrality.pagerank
    elif algo_ == "Eigenvector":
        algo = graph_tool.centrality.eigenvector
    elif algo_ == "hits":
        algo = graph_tool.centrality.hits
    
    baseline = algo(G_dict["original"][0], weight=G_dict["original"][0].edge_properties["weight"])
    if type(baseline) == tuple:
        baseline = baseline[0]
    baseline = list(baseline)
    baseline = np.argsort(baseline)


    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            res = algo(Graph, weight=Graph.edge_properties["weight"])
            if type(res) == tuple:
                res = res[0]
            res = list(res)
            res = np.argsort(res)
            # res = [x[0] for x in res]

            intersect = set(baseline[:topN]).intersection(set(res[:topN]))
            # baseline_ind = [baseline.index(x) for x in intersect]
            # res_ind = [res.index(x) for x in intersect]

            topN_precision = len(intersect) / topN
            topN_correlation = scipy.stats.spearmanr(baseline[:topN], res[:topN])[0]
            
            print(f"{name}\t {Graph}\t top {topN} precision: {topN_precision:.2f}\t top {topN} correlation: {topN_correlation:.2f}")
            # print(f"{name}\t {Graph}")

    t_e, pt_e = time.time(), time.process_time()
    print(f"Time: {t_e-t_s:.2f} s\t Process Time: {pt_e-pt_s:.2f} s")


@timeout(MAX_TIMEOUT)
def GlobalClusteringCoefficient_gt(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"GlobalClusteringCoefficient", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")
        
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            gcc = list(graph_tool.clustering.global_clustering(Graph, weight=Graph.edge_properties["weight"]))
            if logToFile:
                fout.write(f"{name}\t{Graph}\tGlobal_Clustering_Coefficient: {gcc[0]:.3f}\n")
            else:
                print(f"{name}\t{Graph}\tGlobal_Clustering_Coefficient: {gcc[0]:.3f}")


@timeout(MAX_TIMEOUT)
def LocalClusteringCoefficient_gt(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"LocalClusteringCoefficient", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    baseline = list(graph_tool.clustering.local_clustering(G_dict["original"][0], weight=G_dict["original"][0].edge_properties["weight"], undirected=False))
    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            lcc = list(graph_tool.clustering.local_clustering(Graph, weight=Graph.edge_properties["weight"], undirected=False))

            ratio = list(map(lambda x: x[0]/x[1] if x[1]!=0 else np.nan, zip(lcc, baseline)))
            ratio_mean = np.ma.masked_invalid(ratio).mean()
            ratio_min = np.ma.masked_invalid(ratio).min()
            ratio_max = np.ma.masked_invalid(ratio).max()
            ratio_std = np.ma.masked_invalid(ratio).std()

            if logToFile:
                fout.write(f"{name}\t{Graph}\tLocal_Clustering_Coefficient, ratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}, mean Clustering Coefficient: {np.mean(lcc):.3f}\n")
            else:
                print(f"{name}\t{Graph}\tLocal_Clustering_Coefficient, ratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}, mean Clustering Coefficient: {np.mean(lcc):.3f}")
    
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def MaxFlow_gt(dataset_name, G_dict, logToFile=False):
    if logToFile:
        outfile = osp.join(PROJECT_HOME, "output_metric_raw", dataset_name, f"MaxFlow", "log")
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        fout = open(outfile, "w")

    x = np.random.randint(0, G_dict["original"][0].num_vertices(), size=(100, 2))
    x = np.delete(x, np.where(x[:,0] == x[:,1]), axis=0)
    cap = G_dict["original"][0].edge_properties["weight"]
    baseline = []

    for s, d in x:
        # print(s,d)
        src, dst = G_dict["original"][0].vertex(s), G_dict["original"][0].vertex(d)
        res = gt.boykov_kolmogorov_max_flow(G_dict["original"][0], src, dst, cap)
        res.a = cap.a - res.a
        max_flow = sum(res[e] for e in dst.in_edges())
        baseline.append(max_flow)

    for name, Graphs in G_dict.items():
        for Graph in Graphs:
            cap = Graph.edge_properties["weight"]
            max_flows = []
            for s, d in x:
                src, dst = Graph.vertex(s), Graph.vertex(d)
                res = gt.boykov_kolmogorov_max_flow(Graph, src, dst, cap)
                res.a = cap.a - res.a
                max_flow = sum(res[e] for e in dst.in_edges())
                max_flows.append(max_flow)

            # ratio = np.ma.divide(np.array(max_flows), np.array(baseline))
            ratio = list(map(lambda x: x[0]/x[1] if x[0]!=0 and x[1]!=0 and x[1] != float('inf') else float("inf"), zip(max_flows, baseline)))
            ratio_mean = np.ma.masked_invalid(ratio).mean()
            ratio_min = np.ma.masked_invalid(ratio).min()
            ratio_max = np.ma.masked_invalid(ratio).max()
            ratio_std = np.ma.masked_invalid(ratio).std()
            if logToFile:
                fout.write(f"{name}\t#nodes: {Graph.num_vertices()}\t#edges: {len(Graph.get_edges())}\tratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}, unreachable: {ratio.count(float('inf'))/len(ratio):.3f}\n")
            else:
                print(f"{name}\t#nodes: {Graph.num_vertices()}\t#edges: {len(Graph.get_edges())}\tratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}, unreachable: {ratio.count(float('inf'))/len(ratio):.3f}")
    
    if logToFile:
        fout.close()


@timeout(MAX_TIMEOUT)
def min_st_cut_gt(G_gt):
    x = np.random.randint(0, originalGraph_gt.num_vertices(), size=(100, 2))
    x = np.delete(x, np.where(x[:,0] == x[:,1]), axis=0)
    cap = originalGraph_gt.edge_properties["weight"]
    baseline = []
    for s, d in x:
        # print(s,d)
        src, dst = originalGraph_gt.vertex(s), originalGraph_gt.vertex(d)
        res = gt.boykov_kolmogorov_max_flow(originalGraph_gt, src, dst, cap)
        part = gt.min_st_cut(originalGraph_gt, src, cap, res)
        mc = sum([max(cap[e] - res[e], 0) for e in originalGraph_gt.edges() if part[e.source()] != part[e.target()]])
        baseline.append(mc)
    print(baseline)

    for name, Graph in G_gt.items():
        cap = Graph.edge_properties["weight"]
        mcs = []
        for s, d in x:
            src, dst = Graph.vertex(s), Graph.vertex(d)
            res = gt.boykov_kolmogorov_max_flow(Graph, src, dst, cap)
            part = gt.min_st_cut(Graph, src, cap, res)
            mc = sum([max(cap[e] - res[e], 0) for e in Graph.edges() if part[e.source()] != part[e.target()]])
            mcs.append(mc)
        # mcs = list(map(lambda x: round(x, 3) if x > 1e-8 else 0, mcs)) 
        # print(mcs)

        ratio = np.ma.divide(np.array(mcs), np.array(baseline))
        ratio_mean = np.ma.masked_invalid(ratio).mean()
        ratio_min = np.ma.masked_invalid(ratio).min()
        ratio_max = np.ma.masked_invalid(ratio).max()
        ratio_std = np.ma.masked_invalid(ratio).std()
        # gt.graph_draw(Graph, edge_pen_width=gt.prop_to_size(cap, mi=1, ma=10, power=1),
        #       vertex_fill_color=part, output=f"gt_out/{name.replace(' ', '')}_min-st-cut.pdf")
        print(f"{name}\t#nodes: {Graph.num_vertices()}\t#edges: {len(Graph.get_edges())}\t min cut ratio min / max / mean / std: {ratio_min:.3f} / {ratio_max:.3f} / {ratio_mean:.3f} / {ratio_std:.3f}")


@timeout(MAX_TIMEOUT)
def min_cut_gt(G_gt):
    ranks = []
    for name, Graph in G_gt.items():
        Graph.set_directed(False)
        weight = Graph.edge_properties["weight"]
        mc, part = gt.min_cut(Graph, weight)
        ranks.append((name, mc))
        print(f"{name}#nodes {Graph.num_vertices()}\t#edges {len(Graph.get_edges())}\t min cut: {mc}")
    # print rankings
    print("\n-------Ranking:-------\n")
    print("\n".join([f"{r[0]}\t{r[1]:.3f}" for r in sorted(ranks, key=lambda x: x[1])]))

