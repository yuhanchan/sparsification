import graph_tool.all as gt
import numpy as np
import scipy
import matplotlib.pyplot as plt

# with open("data/ErdosRenyi200/raw/uduw.el", 'r') as f:
with open("data/ErdosRenyi200/pruned/er_max/0.8/udw.wel", 'r') as f:
# with open("data/ErdosRenyi200/pruned/random/0.8/uduw.el", 'r') as f:
    el = np.loadtxt(f, dtype=float)
# el = np.hstack((el, np.ones((el.shape[0], 1)).astype(dtype=int)))

g = gt.Graph(directed=False)
eweight = g.new_edge_property("float")
g.add_edge_list(el, eprops=[eweight])
g.properties[("e", "weight")] = eweight

weight = g.edge_properties["weight"]
mc, part = gt.min_cut(g, weight)
print("min cut: ", mc)
gt.graph_draw(g, edge_pen_width=weight, vertex_fill_color=part,
              output="example-min-cut.pdf")

A = gt.laplacian(g, norm=True, operator=True)
N = g.num_vertices()
ew1 = scipy.sparse.linalg.eigs(A, k=N//2, which="LR", return_eigenvectors=False)
ew2 = scipy.sparse.linalg.eigs(A, k=N-N//2, which="SR", return_eigenvectors=False)
ew = np.concatenate((ew1, ew2))

plt.figure(figsize=(8, 2))
plt.scatter(np.real(ew), np.imag(ew), c=np.sqrt(abs(ew)), linewidths=0, alpha=0.6)
plt.xlabel(r"$\operatorname{Re}(\lambda)$")
plt.ylabel(r"$\operatorname{Im}(\lambda)$")
plt.tight_layout()
plt.savefig("laplacian-spectrum.svg")

# print(g.edge_properties)