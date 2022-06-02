from random import randint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nodes", type=int, default=100, help="number of nodes")
parser.add_argument("-e", "--edges", type=int, default=10000, help="number of edges")
parser.add_argument("-o", "--output", type=str, default="toy.in", help="output filename")
args = parser.parse_args()

n = args.nodes

edges = []

old_dst = 0
for i in range(args.edges):
    src, dst = old_dst, randint(0, n)
    old_dst = dst
    if src != dst:
        edges.append(f"{src} {dst}\n")
        edges.append(f"{dst} {src}\n")

# remove duplicates
edges = list(set(edges))

# sort edges by first node, and then by second node
sorted_edges = sorted(edges, key=lambda x: (int(x.split()[0]), int(x.split()[1])))

# print(sorted_edges)

with open(args.output, "w") as f:
    f.writelines(sorted_edges)
