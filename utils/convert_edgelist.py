import json
import os
import multiprocessing
import subprocess
import shlex
from multiprocessing.pool import ThreadPool
from os import path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default=None, required=True)
parser.add_argument("--num_thread", type=int, default=4)
args = parser.parse_args()


PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ") 
    print("please source env.sh at the top level of the project")
    exit(1)


def call_proc(cmd):
    """ This runs in a separate thread. """
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)


# init pool
pool = ThreadPool(args.num_thread)
results = []

config = json.load(open(f"{PROJECT_HOME}/config.json", 'r'))

# Define the names of the datasets to be used.
if args.dataset_name == "all":
    dataset_names = [
        "ego-Facebook",
        "ego-Twitter",
        "soc-Pokec",
        "human_gene2",
        "cage14",
        "com-DBLP",
        "com-LiveJournal",
        "com-Amazon",
        "email-Enron",
        "wiki-Talk",
        "ca-AstroPh",
        "ca-HepPh",
        "web-BerkStan",
        "web-Google",
        "web-NotreDame",
        "web-Stanford",
        "roadNet-CA",
        "Reddit",
        "ogbn-products",
        "ogbn-proteins",
    ]
else:
    dataset_names = [args.dataset_name]


def convert(src, dst, mode):
    if not osp.exists(dst):
        results.append(pool.apply_async(call_proc, (f"{osp.join(PROJECT_HOME, 'utils/bin/utils')} -i {src} -o {dst} -m {mode}",)))


for dataset_name in dataset_names:
    for prune_algo in ["LocalDegree", "LSpar", "GSpar", "LocalSimilarity", "SCAN"]:
        if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            continue
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel", 
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/udw.wel", 
                        mode=3)
            elif config[dataset_name]["directed"]:
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el",
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/uduw.el",
                        mode=1)
            elif config[dataset_name]["weighted"]:
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/udw.wel",
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/dw.wel",
                        mode=4)
            else:
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/uduw.el",
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/0/duw.el",
                        mode=2)

    for prune_algo in ["Random", "KNeighbor", "RankDegree", "ForestFire"]:
        if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            continue
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                if config[dataset_name]["directed"] and config[dataset_name]["weighted"]:
                    convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel",
                            dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel",
                            mode=3)
                elif config[dataset_name]["directed"]:
                    convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el",
                            dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el",
                            mode=1)
                elif config[dataset_name]["weighted"]:
                    convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel",
                            dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel",
                            mode=4)
                else:
                    convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el",
                            dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el",
                            mode=2)

    for prune_algo in ["SpanningForest", "Spanner-3", "Spanner-5", "Spanner-7"]:
        if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            continue
        if config[dataset_name]["weighted"]:
            convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/udw.wel",
                    dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/dw.wel",
                    mode=4)
        else:
            convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/uduw.el",
                    dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/0/duw.el",
                    mode=2)

    for prune_algo in ["ER-Min", "ER-Max"]:
        if not osp.exists(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            continue
        for prune_rate in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}"):
            for run in os.listdir(f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}"):
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/duw.el",
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/uduw.el",
                        mode=1)
                convert(src=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/dw.wel",
                        dst=f"{PROJECT_HOME}/data/{dataset_name}/pruned/{prune_algo}/{prune_rate}/{run}/udw.wel",
                        mode=3)


# Close the pool and wait for each running task to complete
pool.close()
pool.join()
for result in results:
    out, err = result.get()
    print(f"out: {out} err: {err}")