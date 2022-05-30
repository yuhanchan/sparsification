"""
This scipt is used to visualize the graph.
"""

import networkx as nx
import sys
import matplotlib.pyplot as plt

def main(argv):
    """
    Input: edge_file, output_file
    """
    if len(argv) != 3:
        print("Usage: python viz.py edge_file output_file")
        print("Example: python viz.py ring.txt ring.png")
        sys.exit(1)

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
    plt.savefig(f"{argv[2]}")


if __name__ == '__main__':
    main(sys.argv)

