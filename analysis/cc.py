import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def cpu_time(folder, dataset="Reddit"):
    outfile_path = osp.join(
        osp.dirname(osp.realpath(__file__)), "cc", dataset, "cpu_time.txt"
    )
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, "w")
    outfile.write("prune rate: cpu_time (s)\n\n")

    plot_dict = {}

    num_trail = 0
    cpu_time = []
    x = []
    y = []
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, "baseline/stdout.txt")) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "CPU Time" in lines[i]:
                cpu_time.append(float(lines[i].strip().split(" ")[2]))
                num_trail += 1
        cpu_time_mean = np.mean(cpu_time)
        cpu_time_std = np.std(cpu_time)
        cpu_time = [
            x
            for x in cpu_time
            if x < cpu_time_mean + 1.5 * cpu_time_std
            and x > cpu_time_mean - 1.5 * cpu_time_std
        ]
        outfile.write(
            f"baseline: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers)\n\n"
        )
        x.append(0)
        y.append(np.mean(cpu_time))
        plot_dict["baseline"] = [x, y]

    for prune_algo in ["random", "in_degree", "out_degree", "old_er", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            num_trail = 0
            cpu_time = []
            with open(osp.join(folder, dataset, prune_algo, subdir, "stdout.txt")) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if "CPU Time" in lines[i]:
                        cpu_time.append(float(lines[i].strip().split(" ")[2]))
                        num_trail += 1
                cpu_time_mean = np.mean(cpu_time)
                cpu_time_std = np.std(cpu_time)
                cpu_time = [
                    x
                    for x in cpu_time
                    if x < cpu_time_mean + 1.5 * cpu_time_std
                    and x > cpu_time_mean - 1.5 * cpu_time_std
                ]
                outfile.write(
                    f"{subdir}: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers)\n"
                )
                x.append(float(subdir))
                y.append(np.mean(cpu_time))
        plot_dict[prune_algo] = [
            plot_dict["baseline"][0] + x,
            plot_dict["baseline"][1] + y,
        ]
        outfile.write("\n")
    outfile.close()

    del plot_dict["baseline"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[1], label=key)
    ax.set_xlabel("prune rate")
    ax.set_ylabel("cpu time (s)")
    ax.legend()
    ax.set_title(f"CPU time for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), "cpu_time.png"))


def component_count(folder, dataset="Reddit"):
    outfile_path = osp.join(
        osp.dirname(osp.realpath(__file__)), "cc", dataset, "component_count.txt"
    )
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, "w")
    outfile.write("prune rate: component_count\n\n")

    plot_dict = {}

    outfile.write(f"------------------baseline--------------------\n")
    x = []
    y = []
    with open(osp.join(folder, dataset, "baseline/analysis.txt")) as f:
        for line in f:
            if "components" in line:
                count = int(line.strip().split(" ")[2])
        outfile.write(f"baseline: {count}\n\n")
    x.append(0)
    y.append(count)
    plot_dict["baseline"] = [x, y]

    for prune_algo in ["random", "sym_degree", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            with open(
                osp.join(folder, dataset, prune_algo, subdir, "analysis.txt")
            ) as f:
                for line in f:
                    if "components" in line:
                        count = int(line.strip().split(" ")[2])
                outfile.write(f"{subdir}: {count}\n")
                x.append(float(subdir))
                y.append(count)
        plot_dict[prune_algo] = [
            plot_dict["baseline"][0] + x,
            plot_dict["baseline"][1] + y,
        ]
        outfile.write("\n")

    del plot_dict["baseline"]
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[1], label=key)
    ax.set_xlabel("prune rate")
    ax.set_ylabel("component count")
    ax.legend()
    ax.set_title(f"Component count for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), "compenent_count.png"))


if __name__ == "__main__":
    # cpu_time("../experiments/cc/", dataset="Reddit")
    # cpu_time("../experiments/cc/", dataset="Reddit2")
    # cpu_time('../experiments/cc/', dataset='ogbn_products')

    # component_count("../experiments/cc/", dataset="Reddit")
    # component_count("../experiments/cc/", dataset="Reddit2")
    component_count("../experiments/cc/", dataset="ogbn_products")
