import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def cpu_time(folder, dataset="Reddit"):
    outfile_path = osp.join(osp.dirname(osp.realpath(__file__)), "cc_sv", dataset, 'cpu_time.txt')
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, 'w')
    outfile.write('prune rate: cpu_time (s)\n\n')
    
    num_trail = 0
    cpu_time = []
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, 'baseline/stdout.txt')) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'CPU Time' in lines[i]:
                cpu_time.append(float(lines[i].strip().split(' ')[2]))
                num_trail += 1
        cpu_time_mean = np.mean(cpu_time)
        cpu_time_std = np.std(cpu_time)
        cpu_time = [x for x in cpu_time if x < cpu_time_mean + 1.5 * cpu_time_std and x > cpu_time_mean - 1.5 * cpu_time_std]
        outfile.write(f"baseline: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers)\n\n")
    
    for prune_algo in ["random", "in_degree", "out_degree", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            num_trail = 0
            cpu_time = []
            with open(osp.join(folder, dataset, prune_algo, subdir, 'stdout.txt')) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if 'CPU Time' in lines[i]:
                        cpu_time.append(float(lines[i].strip().split(' ')[2]))
                        num_trail += 1
                cpu_time_mean = np.mean(cpu_time)
                cpu_time_std = np.std(cpu_time)
                cpu_time = [x for x in cpu_time if x < cpu_time_mean + 1.5 * cpu_time_std and x > cpu_time_mean - 1.5 * cpu_time_std]
                outfile.write(f"{subdir}: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers)\n")
        outfile.write("\n")
        
    outfile.close()


def component_count(folder, dataset="Reddit"):
    outfile_path = osp.join(osp.dirname(osp.realpath(__file__)), "cc_sv", dataset, 'component_count.txt')
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, 'w')
    outfile.write('prune rate: component_count\n\n')
    
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, 'baseline/analysis.txt')) as f:
        for line in f:
            if 'components' in line:
                count = line.strip().split(' ')[2]
        outfile.write(f"baseline: {count}\n\n")
    
    for prune_algo in ["random", "in_degree", "out_degree", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            with open(osp.join(folder, dataset, prune_algo, subdir, 'analysis.txt')) as f:
                for line in f:
                    if 'components' in line:
                        count = line.strip().split(' ')[2]
                outfile.write(f"{subdir}: {count}\n")
        outfile.write("\n") 

if __name__ == "__main__":
    cpu_time('../experiments/cc_sv/', dataset='Reddit')
    cpu_time('../experiments/cc_sv/', dataset='Reddit2')
    cpu_time('../experiments/cc_sv/', dataset='ogbn_products')
    
    component_count('../experiments/cc_sv/', dataset='Reddit')
    component_count('../experiments/cc_sv/', dataset='Reddit2')
    component_count('../experiments/cc_sv/', dataset='ogbn_products')
    