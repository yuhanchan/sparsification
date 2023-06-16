#####
# Examples: python parse.py -i email-Enron.txt
# ######

import glob
from collections import defaultdict
from collections import OrderedDict
import os
import re
import os.path as osp
import argparse
parser = argparse.ArgumentParser()

PROJECT_HOME = os.getenv(key="PROJECT_HOME")
if PROJECT_HOME is None:
    print("PROJECT_HOME is not set, ")
    print("please source env.sh at the top level of the project")
    exit(1)

# parser.add_argument('-i', required=True)
# args = parser.parse_args()

class graph_stats:
    graph_name = None
    num_node = -1
    num_edge = -1
    directed = None
    weighted = None
    isolated_node = -1
    self_loops = -1
    density = -1
    cluster_coeff = -1
    min_degree = -1
    max_degree = -1
    avg_degree = -1
    degree_assort = -1
    num_cc = -1
    sz_largest_cc = -1
    consumed_mem = -1
    wall_time = -1
    cpu_time = -1
    def __repr__(self):
        return (self.graph_name + "," + \
                # str(self.num_node) + "," + \
                str(self.num_edge) + "," + \
                # self.directed + ";" + \
                # self.weighted + ";" + \
                # str(self.isolated_node) + ";" + \
                # str(self.self_loops) + ";" + \
                # str(self.density) + ";" + \
                # str(self.cluster_coeff) + ";" + \
                # str(self.min_degree) + ";" + \
                # str(self.max_degree) + ";" + \
                # str(self.avg_degree) + ";" + \
                # str(self.degree_assort) + ";" + \
                # str(self.num_cc) + ";" + \
                # str(self.sz_largest_cc) + ";" + \
                str(self.consumed_mem) + "," + \
                str(self.wall_time) + "," + \
                str(self.cpu_time)
        )


def parse(dataset_name):
    g = graph_stats()
    g.graph_name = "Full"
    f = open(osp.join(PROJECT_HOME, f"output_sparsifier_raw/{dataset_name}.txt"), "r")
    os.makedirs(osp.join(PROJECT_HOME, f"output_sparsifier_parsed"), exist_ok=True)
    fout = open(osp.join(PROJECT_HOME, f"output_sparsifier_parsed/{dataset_name}.csv"), "w")
    fout.write("prune_algo,num_edge,consumed_mem,wall_time,cpu_time\n")
    for line in f.readlines():
        if ":consumed memory:" in line:
            g.graph_name = line.split(':')[0]
            g.consumed_mem = float(re.sub(",","",line.split()[2]))
            g.wall_time = str(float(line.split()[5]))
            g.cpu_time = str(float(line.split()[9]))
            if g.num_edge != -1:
                # print(g)
                fout.write(str(g) + "\n")
            g = graph_stats()
        
        if "nodes, edges" in line:
            g.num_node = line.split()[-2].replace(",","")
            g.num_edge = line.split()[-1]
        
        if "Prune rate for epsilon" in line:
            g.num_edge = line.split()[-1].replace(")","")

        # if "directed?" in line:
        #     g.directed = line.split()[-1]

        # if "weighted?" in line:
        #     g.weighted = line.split()[-1]

        # if "isolated nodes" in line:
        #     g.isolated_node = line.split()[-1]

        # if "self-loops" in line:
        #     g.self_loops = line.split()[-1]

        # if "density" in line:
        #     g.density = line.split()[-1]

        # if "clustering coefficient" in line:
        #     g.cluster_coeff = line.split()[-1]

        # if "min/max/avg degree" in line:
        #     g.min_degree = line.split()[-3].split(',')[0]
        #     g.max_degree = line.split()[-2].split(',')[0]
        #     g.avg_degree = line.split()[-1]

        # if "degree assortativity" in line:
        #     g.degree_assort = line.split()[-1]

        # if "number of connected components" in line:
        #     g.num_cc = line.split()[-1]

        # if "size of largest component" in line:
        #     g.sz_largest_cc = line.split()[4]

        # if ":" in line and "Network Properties:" not in line and \
        #             ":consumed memory:" not in line:
        #     g = graph_stats()
        #     g.graph_name = line.split(':')[0]

    f.close()
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()

    if args.dataset_name == "all":
        dataset_names = ["ego-Facebook",
                        "ego-Twitter", 
                        "soc-Pokec", 
                        "human_gene2", 
                        "cage14", 
                        "com-DBLP", 
                        "com-LiveJournal", 
                        "com-Amazon", 
                        "email-Enron", 
                        "wiki-Talk", 
                        "cit-HepPh", 
                        "ca-AstroPh", 
                        "ca-HepPh", 
                        "web-BerkStan", 
                        "web-Google", 
                        "web-NotreDame", 
                        "web-Stanford", 
                        "roadNet-CA", 
                        "Reddit", 
                        "ogbn-products", 
                        "ogbn-proteins"]
    else:
        dataset_names = [args.dataset_name]

    for dataset_name in dataset_names:
        parse(dataset_name)