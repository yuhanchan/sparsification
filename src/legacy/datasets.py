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


def randomGraphGenerator():
    # gen graph
    # Initalize algorithm
    dataset_name = "ErdosRenyi200"
    originalGraph = nk.generators.ErdosRenyiGenerator(200, 0.2)
    # originalGraph = nk.generators.BarabasiAlbertGenerator(10, 200)
    # originalGraph = nk.generators.DorogovtsevMendesGenerator(200)
    # Run algorithm
    originalGraph = originalGraph.generate()
    output_file = f"data/{dataset_name}/raw/uduw.el"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    nk.writeGraph(originalGraph, output_file, nk.Format.EdgeListSpaceZero)
    originalGraph = nk.readGraph(f"data/{dataset_name}/raw/uduw.el", nk.Format.EdgeListSpaceZero)
