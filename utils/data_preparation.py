import os
import argparse
import os.path as osp
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.datasets as PygDataset
import torch
import numpy as np

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help="dataset_name, or 'all'")
args = parser.parse_args()


if args.dataset_name == "all":
    dataset_names = [
        "ego-Facebook",
        "ego-Twitter",
        "human_gene2",
        "com-DBLP",
        "com-Amazon",
        "email-Enron",
        "wiki-Talk",
        "ca-AstroPh",
        "ca-HepPh",
        "web-BerkStan",
        "web-Google",
        "web-NotreDame",
        "web-Stanford",
        "Reddit",
        "ogbn-proteins",
    ]
else:
    dataset_names = [args.dataset_name]


for dataset_name in dataset_names:
    if dataset_name == "ego-Facebook":
        os.system(f"curl https://snap.stanford.edu/data/facebook_combined.txt.gz --create-dirs -o {PROJECT_HOME}/data/ego-Facebook/raw/facebook_combined.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/ego-Facebook/raw/facebook_combined.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/ego-Facebook/raw/facebook_combined.txt 0 0")
    if dataset_name == "ego-Twitter":
        os.system(f"curl https://snap.stanford.edu/data/twitter_combined.txt.gz --create-dirs -o {PROJECT_HOME}/data/ego-Twitter/raw/twitter_combined.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/ego-Twitter/raw/twitter_combined.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/ego-Twitter/raw/twitter_combined.txt 1 0")
    if dataset_name == "human_gene2":
        os.system(f"curl https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/human_gene2.tar.gz --create-dirs -o {PROJECT_HOME}/data/human_gene2/raw/human_gene2.txt.gz")
        os.system(f"tar -xvf {PROJECT_HOME}/data/human_gene2/raw/human_gene2.tar.gz -C {PROJECT_HOME}/data/human_gene2/raw/")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/human_gene2/raw/human_gene2.txt 0 1")
    if dataset_name == "com-DBLP":
        os.system(f"curl https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz --create-dirs -o {PROJECT_HOME}/data/com-DBLP/raw/com-dblp.ungraph.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/com-DBLP/raw/com-dblp.ungraph.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/com-DBLP/raw/com-dblp.ungraph.txt 0 0")
    if dataset_name == "com-Amazon":
        os.system(f"curl https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz --create-dirs -o {PROJECT_HOME}/data/com-Amazon/raw/com-amazon.ungraph.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/com-Amazon/raw/com-amazon.ungraph.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/com-Amazon/raw/com-amazon.ungraph.txt 0 0")
    if dataset_name == "email-Enron":
        os.system(f"curl https://snap.stanford.edu/data/email-Enron.txt.gz --create-dirs -o {PROJECT_HOME}/data/email-Enron/raw/email-Enron.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/email-Enron/raw/email-Enron.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/email-Enron/raw/email-Enron.txt 0 0")
    if dataset_name == "wiki-Talk":
        os.system(f"curl https://snap.stanford.edu/data/wiki-Talk.txt.gz --create-dirs -o {PROJECT_HOME}/data/wiki-Talk/raw/wiki-Talk.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/wiki-Talk/raw/wiki-Talk.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/wiki-Talk/raw/wiki-Talk.txt 1 0")
    if dataset_name == "ca-AstroPh":
        os.system(f"curl https://snap.stanford.edu/data/ca-AstroPh.txt.gz --create-dirs -o {PROJECT_HOME}/data/ca-AstroPh/raw/ca-AstroPh.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/ca-AstroPh/raw/ca-AstroPh.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/ca-AstroPh/raw/ca-AstroPh.txt 0 0")
    if dataset_name == "ca-HepPh":
        os.system(f"curl https://snap.stanford.edu/data/ca-HepPh.txt.gz --create-dirs -o {PROJECT_HOME}/data/ca-HepPh/raw/ca-HepPh.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/ca-HepPh/raw/ca-HepPh.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/ca-HepPh/raw/ca-HepPh.txt 0 0")
    if dataset_name == "web-BerkStan":
        os.system(f"curl https://snap.stanford.edu/data/web-BerkStan.txt.gz --create-dirs -o {PROJECT_HOME}/data/web-BerkStan/raw/web-BerkStan.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/web-BerkStan/raw/web-BerkStan.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/web-BerkStan/raw/web-BerkStan.txt 1 0")
    if dataset_name == "web-Google":
        os.system(f"curl https://snap.stanford.edu/data/web-Google.txt.gz --create-dirs -o {PROJECT_HOME}/data/web-Google/raw/web-Google.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/web-Google/raw/web-Google.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/web-Google/raw/web-Google.txt 1 0")
    if dataset_name == "web-NotreDame":
        os.system(f"curl https://snap.stanford.edu/data/web-NotreDame.txt.gz --create-dirs -o {PROJECT_HOME}/data/web-NotreDame/raw/web-NotreDame.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/web-NotreDame/raw/web-NotreDame.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/web-NotreDame/raw/web-NotreDame.txt 1 0")
    if dataset_name == "web-Stanford":
        os.system(f"curl https://snap.stanford.edu/data/web-Stanford.txt.gz --create-dirs -o {PROJECT_HOME}/data/web-Stanford/raw/web-Stanford.txt.gz")
        os.system(f"gzip -d {PROJECT_HOME}/data/web-Stanford/raw/web-Stanford.txt.gz")
        os.system(f"bash {PROJECT_HOME}/utils/data_preparation.sh {PROJECT_HOME}/data/web-Stanford/raw/web-Stanford.txt 1 0")
    if dataset_name == "Reddit":
        dataset = PygDataset.Reddit(f"{PROJECT_HOME}/data/Reddit")
        # save test_data.data.edge_index to file
        np.savetxt(osp.join(PROJECT_HOME, f"data/Reddit/raw/duw.el"), 
                   np.transpose(test_data.data.edge_index.numpy()), fmt="%d")
    if dataset_name == "ogbn-proteins":
        dataset = PygNodePropPredDataset(
            "ogbn-proteins",
            root=osp.join(PROJECT_HOME, "data"),
            transform=T.ToSparseTensor(attr="edge_attr"),
        )
        # save adj_t to file
        np.savetxt(osp.join(PROJECT_HOME, f"data/ogbn-proteins/raw/duw.el"), 
                   np.transpose(dataset.data.edge_index.numpy()), fmt="%d")

