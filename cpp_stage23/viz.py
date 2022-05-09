import networkx as nx
import sys
import matplotlib.pyplot as plt

def main(argv):
    # undirected graph
    G = nx.Graph()

    # read in the edges
    with open(argv[1], 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip().split()
            if line:
                G.add_edge(int(line[0]), int(line[1]))

    # save the graph to png
    nx.draw(G, with_labels=True, node_size=100)
    plt.savefig("viz.png")


if __name__ == '__main__':
    main(sys.argv)

