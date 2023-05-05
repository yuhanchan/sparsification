from graph_tool.all import *
import graph_tool.all as gt
import numpy as np
import scipy
import matplotlib.pyplot as plt

# g = Graph()

# g.add_vertex(232965)

# with open("../data/Reddit2/raw/duw.el", "r") as f:
#     for line in f:
#         src, dst = line.strip().split(" ")
#         src = int(src)
#         dst = int(dst)
#         g.add_edge(src, dst)

# graph_draw(g, vertex_text=g.vertex_index, output="two-nodes.pdf")

srcs = []
dsts = []
N = 0
# with open("../data/gd/raw/duw.el", "r") as f:
with open("../data/gd/pruned/python_er/", "r") as f:
    lines = f.readlines()
    for line in lines:
        src, dst = line.strip().split(" ")
        src = int(src)
        dst = int(dst)
        srcs.append(src)
        dsts.append(dst)
        N = max(N, src, dst)

g = Graph()
g.add_vertex(N)
for i in range(len(srcs)):
    g.add_edge(srcs[i], dsts[i])

# g = gt.collection.data["polblogs"]
A = gt.adjacency(g, operator=True)
N = g.num_vertices()
ew1 = scipy.sparse.linalg.eigs(A, k=N // 2, which="LR", return_eigenvectors=False)
ew2 = scipy.sparse.linalg.eigs(A, k=N - N // 2, which="SR", return_eigenvectors=False)
ew = np.concatenate((ew1, ew2))

plt.figure(figsize=(8, 2))
plt.scatter(np.real(ew), np.imag(ew), c=np.sqrt(abs(ew)), linewidths=0, alpha=0.6)
plt.xlabel(r"$\operatorname{Re}(\lambda)$")
plt.ylabel(r"$\operatorname{Im}(\lambda)$")
plt.tight_layout()
plt.savefig("laplacian-spectrum.svg")
