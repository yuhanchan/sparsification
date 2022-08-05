import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def cpu_time(folder, dataset="Reddit"):
    outfile_path = osp.join(
        osp.dirname(osp.realpath(__file__)), "pr", dataset, "cpu_time.txt"
    )
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, "w")
    outfile.write(
        "prune rate: cpu_time (s), iteration to converge, cpu_time/iteration (s)\n\n"
    )

    plot_dict = {}

    num_trail = 0
    cpu_time = []
    x = []
    y = []
    total_iteration = 0
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, "baseline/stdout.txt")) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "CPU Time" in lines[i]:
                cpu_time.append(float(lines[i].strip().split(" ")[2]))
                num_trail += 1
                total_iteration += int(lines[i - 1].strip().split(" ")[0])
        cpu_time_mean = np.mean(cpu_time)
        cpu_time_std = np.std(cpu_time)
        cpu_time = [
            x
            for x in cpu_time
            if x < cpu_time_mean + 1.5 * cpu_time_std
            and x > cpu_time_mean - 1.5 * cpu_time_std
        ]
        outfile.write(
            f"baseline: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers), {total_iteration / num_trail}, {np.mean(cpu_time)/(total_iteration/num_trail):.4f}\n\n"
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
            total_iteration = 0
            with open(osp.join(folder, dataset, prune_algo, subdir, "stdout.txt")) as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if "CPU Time" in lines[i]:
                        cpu_time.append(float(lines[i].strip().split(" ")[2]))
                        num_trail += 1
                        total_iteration += int(lines[i - 1].strip().split(" ")[0])
                cpu_time_mean = np.mean(cpu_time)
                cpu_time_std = np.std(cpu_time)
                cpu_time = [
                    x
                    for x in cpu_time
                    if x < cpu_time_mean + 1.5 * cpu_time_std
                    and x > cpu_time_mean - 1.5 * cpu_time_std
                ]
                outfile.write(
                    f"{subdir}: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers), {total_iteration / num_trail}, {np.mean(cpu_time)/(total_iteration/num_trail):.4f}\n"
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


def relative_score_error(folder, dataset="Reddit", prune_algo="random"):
    baseline_score = []
    with open(osp.join(folder, dataset, "baseline/analysis.txt")) as f:
        for line in f:
            line = line.strip().split(":")
            node_id, score = line[0], float(line[1])
            baseline_score.append(score)

    error_dict = {}
    for subdir in os.listdir(osp.join(folder, dataset, prune_algo)):
        score_tmp = []
        with open(osp.join(folder, dataset, prune_algo, subdir, "analysis.txt")) as f:
            for line in f:
                line = line.strip().split(":")
                node_id, score = line[0], float(line[1])
                score_tmp.append(score)
            score_sum = sum(score_tmp)
            for i in range(len(score_tmp)):
                score_tmp[i] = score_tmp[i] / score_sum

            error_tmp = []
            for i in range(len(score_tmp)):
                error = (score_tmp[i] - baseline_score[i]) / baseline_score[i]
                error_tmp.append(error)
            error_dict[float(subdir)] = error_tmp
            # print(subdir, sum(score_tmp))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("Node ID")
    ax.set_ylabel("error")
    ax.set_title(f"{dataset} {prune_algo}")

    for key in ["0.8", "0.7", "0.5", "0.3", "0.1"]:
        ax.plot(error_dict[key], label=f"Prune rate {key}")
    ax.legend()

    save_path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "pr",
        dataset,
        prune_algo,
        "relative_score_error.png",
    )
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def precision_k(folder, percent, dataset="Reddit", prune_algo="random"):
    baseline_score = []
    with open(osp.join(folder, dataset, "baseline/analysis.txt")) as f:
        for line in f:
            line = line.strip().split(":")
            node_id, score = line[0], float(line[1])
            baseline_score.append((node_id, score))

    baseline_score = [
        i for i, v in sorted(baseline_score, key=operator.itemgetter(1), reverse=True)
    ]

    score_dict = {}
    for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
        score_tmp = []
        with open(osp.join(folder, dataset, prune_algo, subdir, "analysis.txt")) as f:
            for line in f:
                line = line.strip().split(":")
                node_id, score = line[0], float(line[1])
                score_tmp.append((node_id, score))
        score_tmp = [
            i for i, v in sorted(score_tmp, key=operator.itemgetter(1), reverse=True)
        ]
        score_dict[float(subdir)] = score_tmp

    fig, ax = plt.subplots(figsize=(5, 5))
    for p in percent:
        N = int(len(baseline_score) * p / 100)
        baseline_topN = sorted(baseline_score[:N], reverse=False)
        x = []
        y = []
        for key in score_dict.keys():
            topN = sorted(score_dict[key][:N], reverse=False)
            num_in_topN = 0
            i, j = 0, 0
            while i < N and j < N:
                if baseline_topN[i] == topN[j]:
                    num_in_topN += 1
                    i += 1
                    j += 1
                elif baseline_topN[i] < topN[j]:
                    i += 1
                else:
                    j += 1
            x.append(key)
            y.append(num_in_topN / N)
        ax.plot(x, y, label=f"Top {p}% ({N})")
    ax.legend()
    ax.set_xlabel("prune_rate")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision K for {dataset} {prune_algo}")
    ax.set_ylim(0, 1)

    save_path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "pr",
        dataset,
        prune_algo,
        "precision_k.png",
    )
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)


def precision_k_2(folder, l, percent=False, dataset="Reddit"):
    """
    Similar to precision_k, but plot all pruning algorithms in one figure

    Parameters: 
        folder: folder containing the results of the experiments
        l: a list of N or percentage
        percent: if True, l is a list of percentage, otherwise l is a list of N
        dataset: name of the dataset

    Returns: None 
        Will save to precision_k.png
    """

    baseline_score = []
    with open(osp.join(folder, dataset, "baseline/analysis.txt")) as f:
        for line in f:
            line = line.strip().split(":")
            node_id, score = line[0], float(line[1])
            baseline_score.append((node_id, score))

    # sort baseline_score by score
    baseline_score = [
        i for i, v in sorted(baseline_score, key=operator.itemgetter(1), reverse=True)
    ]

    for p in l:
        fig, ax = plt.subplots(figsize=(5, 5))

        for prune_algo in ["random", "in_degree", "out_degree", "er"]:
            score_dict = {}
            for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
                score_tmp = []
                with open(osp.join(folder, dataset, prune_algo, subdir, "analysis.txt")) as f:
                    for line in f:
                        line = line.strip().split(":")
                        node_id, score = line[0], float(line[1])
                        score_tmp.append((node_id, score))
                score_tmp = [
                    i for i, v in sorted(score_tmp, key=operator.itemgetter(1), reverse=True)
                ]
                score_dict[float(subdir)] = score_tmp


            if percent:
                N = int(len(baseline_score) * p / 100)
            else:
                N = int(p)
            baseline_topN = sorted(baseline_score[:N], reverse=False)
            x = []
            y = []
            for key in score_dict.keys():
                topN = sorted(score_dict[key][:N], reverse=False)
                num_in_topN = 0
                i, j = 0, 0
                while i < N and j < N:
                    if baseline_topN[i] == topN[j]:
                        num_in_topN += 1
                        i += 1
                        j += 1
                    elif baseline_topN[i] < topN[j]:
                        i += 1
                    else:
                        j += 1
                x.append(key)
                y.append(num_in_topN / N)

            ax.plot(x, y, marker='o', label=f"{prune_algo}")

        ax.legend()
        ax.set_xlabel("prune_rate")
        ax.set_ylabel("precision")
        if percent:
            ax.set_title(f"Top {p}% ({N}) Precision K for {dataset} {prune_algo}")
        else:
            ax.set_title(f"Top {N} Precision K for {dataset} {prune_algo}")
        ax.set_ylim(0, 1)

        save_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "pr",
            dataset,
            f"top_{N}_precision_k.png",
        )
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)


def main():
    # cpu_time("../experiments/pr/", dataset="Reddit")
    # cpu_time("../experiments/pr/", dataset="Reddit2")
    # cpu_time('../experiments/pr/', dataset='ogbn_products')

    # relative_score_error(folder='../experiments/pr/', dataset='Reddit', prune_algo='random')
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = {}
    #     # create jobs
    #     # for ds in ['Reddit', 'Reddit2', 'ogbn_products']:
    #     for ds in ["Reddit", "Reddit2"]:
    #         for pa in ["random", "in_degree", "out_degree", "old_er", "er"]:

    #             # futures[executor.submit(relative_score_error, '../experiments/pr', ds, pa)] = f"dataset={ds}, prune_algo={pa}, metric=relative_score_error"
    #             futures[
    #                 executor.submit(
    #                     precision_k,
    #                     "../experiments/pr",
    #                     [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
    #                     ds,
    #                     pa,
    #                 )
    #             ] = f"dataset={ds}, prune_algo={pa}, metric=precision_k"
    #     # run jobs
    #     for future in futures:
    #         print(futures[future])
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(futures[future], e)
    #             sys.exit(1)

    # precision_k_2("../experiments/pr", [3, 5, 10, 15, 20], percent=False, dataset="Reddit")
    # precision_k_2("../experiments/pr", [3, 5, 10, 15, 20], percent=False, dataset="Reddit2")
    precision_k_2("../experiments/bc", [3, 5, 10, 15, 20], percent=False, dataset="ogbn_products")

if __name__ == "__main__":
    main()
