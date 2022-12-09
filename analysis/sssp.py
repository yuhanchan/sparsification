import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
import operator
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def cpu_time(folder, dataset="Reddit"):
    outfile_path = osp.join(
        osp.dirname(osp.realpath(__file__)), "sssp", dataset, "cpu_time.txt"
    )
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, "w")
    outfile.write(
        "prune rate: cpu_time (s), iteration to converge, cpu_time/iteration (s)\n\n"
    )

    plot_dict = {}

    num_trail = 0
    cpu_time = []
    total_iteration = 0
    x = []
    y = []
    y1 = []
    y2 = []
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, "baseline/stdout.txt")) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "CPU Time" in lines[i]:
                cpu_time.append(float(lines[i].strip().split(" ")[2]))
                num_trail += 1
                total_iteration += int(lines[i - 1].strip().split(" ")[1])
        cpu_time_mean = np.mean(cpu_time)
        cpu_time_std = np.std(cpu_time)
        cpu_time = [
            x
            for x in cpu_time
            if x < cpu_time_mean + 1.5 * cpu_time_std
            and x > cpu_time_mean - 1.5 * cpu_time_std
        ]
        outfile.write(
            f"baseline: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers), {total_iteration / num_trail}, {np.mean(cpu_time)/(total_iteration/num_trail):.6f}\n\n"
        )
        x.append(0)
        y.append(np.mean(cpu_time))
        y1.append(total_iteration / num_trail)
        y2.append(np.mean(cpu_time) / (total_iteration / num_trail))
        plot_dict["baseline"] = [x, y, y1, y2]

    for prune_algo in ["random", "in_degree", "out_degree", "old_er", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        y1 = []
        y2 = []
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
                        total_iteration += int(lines[i - 1].strip().split(" ")[1])
                cpu_time_mean = np.mean(cpu_time)
                cpu_time_std = np.std(cpu_time)
                cpu_time = [
                    x
                    for x in cpu_time
                    if x < cpu_time_mean + 1.5 * cpu_time_std
                    and x > cpu_time_mean - 1.5 * cpu_time_std
                ]
                outfile.write(
                    f"{subdir}: {np.mean(cpu_time) :.2f} ({num_trail - len(cpu_time)} Outliers), {total_iteration / num_trail}, {np.mean(cpu_time)/(total_iteration/num_trail):.6f}\n"
                )
                x.append(float(subdir))
                y.append(np.mean(cpu_time))
                y1.append(total_iteration / num_trail)
                y2.append(np.mean(cpu_time) / (total_iteration / num_trail))
        plot_dict[prune_algo] = [
            plot_dict["baseline"][0] + x,
            plot_dict["baseline"][1] + y,
            plot_dict["baseline"][2] + y1,
            plot_dict["baseline"][3] + y2,
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

    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[2], label=key)
    ax.set_xlabel("prune rate")
    ax.set_ylabel("#iteration to converge")
    ax.legend()
    ax.set_title(f"#iteration for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), "iteration.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in plot_dict.items():
        ax.plot(value[0], value[3], label=key)
    ax.set_xlabel("prune rate")
    ax.set_ylabel("cpu time per iteration (s)")
    ax.legend()
    ax.set_title(f"CPU time per iteration for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), "cpu_time_per_iteration.png"))


def reachability(folder, dataset="Reddit"):
    outfile_path = osp.join(
        osp.dirname(osp.realpath(__file__)), "sssp", dataset, "reachability.txt"
    )
    os.makedirs(osp.dirname(outfile_path), exist_ok=True)
    outfile = open(outfile_path, "w")
    outfile.write("prune rate: #Reachable_Nodes/#Nodes (percent)\n\n")

    plot_dict = {}

    total_num_nodes = 0
    reachable_num_nodes = 0
    x = []
    y = []
    outfile.write(f"------------------baseline--------------------\n")
    with open(osp.join(folder, dataset, "baseline/stdout.txt")) as f:
        for line in f:
            if "Graph has" in line:
                line = line.strip().split(" ")
                total_num_nodes = int(line[2])
                break
    with open(osp.join(folder, dataset, "baseline/analysis.txt")) as f:
        line = f.readline().strip().split(" ")
        reachable_num_nodes = int(line[3])
    outfile.write(
        f"baseline: {reachable_num_nodes}/{total_num_nodes} ({reachable_num_nodes/total_num_nodes*100:.2f}%)\n\n"
    )
    x.append(0)
    y.append(reachable_num_nodes / total_num_nodes * 100)
    plot_dict["baseline"] = [x, y]

    for prune_algo in ["random", "in_degree", "out_degree", "old_er", "er"]:
        outfile.write(f"------------------{prune_algo}--------------------\n")
        x = []
        y = []
        for subdir in sorted(os.listdir(osp.join(folder, dataset, prune_algo))):
            with open(osp.join(folder, dataset, prune_algo, subdir, "stdout.txt")) as f:
                for line in f:
                    if "Graph has" in line:
                        line = line.strip().split(" ")
                        total_num_nodes = int(line[2])
                        break
            with open(
                osp.join(folder, dataset, prune_algo, subdir, "analysis.txt")
            ) as f:
                line = f.readline().strip().split(" ")
                reachable_num_nodes = int(line[3])
                outfile.write(
                    f"{subdir}: {reachable_num_nodes}/{total_num_nodes} ({reachable_num_nodes/total_num_nodes*100:.2f}%)\n"
                )
                x.append(float(subdir))
                y.append(reachable_num_nodes / total_num_nodes * 100)
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
    ax.set_ylabel("rechaability (percent)")
    ax.legend()
    ax.set_title(f"Reachability for {dataset}")
    fig.savefig(osp.join(osp.dirname(outfile_path), "reachability.png"))


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

            error_tmp = []
            for i in range(len(score_tmp)):
                if baseline_score[i] == 0:
                    error_tmp.append(0)
                elif score_tmp[i] == -1:
                    error_tmp.append(-1)
                else:
                    error = (score_tmp[i] - baseline_score[i]) / baseline_score[i]
                    error_tmp.append(error)
            error_dict[float(subdir)] = error_tmp
            # print(subdir, sum(score_tmp))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("Node ID")
    ax.set_ylabel("error")
    ax.set_title(f"{dataset} {prune_algo}")

    for key in [0.8, 0.7, 0.5, 0.3, 0.1]:
        ax.plot(error_dict[key], label=f"Prune rate {key}")
        print(key, np.mean(error_dict[key]))
    ax.legend()

    save_path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "sssp",
        dataset,
        prune_algo,
        "relative_score_error.png",
    )
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def relative_score_error_hist(folder, dataset="Reddit", prune_algo="random"):
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

            error_tmp = []
            for i in range(len(score_tmp)):
                if baseline_score[i] == 0:
                    error_tmp.append(0)
                elif score_tmp[i] == -1:
                    error_tmp.append(-1)
                else:
                    error = (score_tmp[i] - baseline_score[i]) / baseline_score[i]
                    error_tmp.append(error)
            error_dict[float(subdir)] = error_tmp
            # print(subdir, sum(score_tmp))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("relative score error")
    ax.set_ylabel("frequency")
    ax.set_title(f"{dataset} {prune_algo}")
    ax.set_xlim(-1, 14)

    # for key in [0.8, 0.7, 0.5, 0.3, 0.1]:
    for key in [0.8]:
        ax.hist(error_dict[key], bins=20, label=f"Prune rate {key}")
        print(key, np.mean(error_dict[key]))
    ax.legend()

    save_path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "sssp",
        dataset,
        prune_algo,
        "relative_score_error_hist.png",
    )
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def main():
    # cpu_time("../experiments/sssp/", dataset="Reddit")
    # cpu_time("../experiments/sssp/", dataset="Reddit2")
    # cpu_time('../experiments/sssp/', dataset='ogbn_products')

    # reachability('../experiments/sssp/', dataset='Reddit')
    # reachability('../experiments/sssp/', dataset='Reddit2')
    # reachability('../experiments/sssp/', dataset='ogbn_products')

    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     futures = {}
    #     # create jobs
    #     for ds in ["Reddit", "Reddit2", "ogbn_products"]:
    #         for pa in ["random", "in_degree", "out_degree", "old_er", "er"]:

    #             # futures[executor.submit(relative_score_error, '../experiments/pr', ds, pa)] = f"dataset={ds}, prune_algo={pa}, metric=relative_score_error"
    #             futures[
    #                 executor.submit(relative_score_error, "../experiments/sssp", ds, pa)
    #             ] = f"dataset={ds}, prune_algo={pa}, metric=relative_score_error"
    #     # run jobs
    #     for future in futures:
    #         print(futures[future])
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(futures[future], e)
    #             sys.exit(1)

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {}
        # create jobs
        # for ds in ["Reddit", "Reddit2", "ogbn_products"]:
        for ds in ["ogbn_products"]:
            for pa in ["random", "sym_degree", "er"]:

                # futures[executor.submit(relative_score_error, '../experiments/pr', ds, pa)] = f"dataset={ds}, prune_algo={pa}, metric=relative_score_error"
                futures[
                    executor.submit(relative_score_error, "../experiments/sssp", ds, pa)
                ] = f"dataset={ds}, prune_algo={pa}, metric=relative_score_error"
        # run jobs
        for future in futures:
            print(futures[future])
            try:
                future.result()
            except Exception as e:
                print(futures[future], e)
                sys.exit(1)


if __name__ == "__main__":
    main()
