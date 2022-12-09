import numpy as np
from itertools import compress

def cos_distance(path1, path2, mapping):
    """ 
    Compare the cosine distance of embeddings from two file paths
    """
    # first row of each file is metadata
    emb1 = np.loadtxt(path1, skiprows=1)
    emb2 = np.loadtxt(path2, skiprows=1)
    # print(emb1.shape, emb2.shape)

    # sort emb1 and emb2 by the first columns
    emb1 = emb1[np.argsort(emb1[:, 0])]
    emb2 = emb2[np.argsort(emb2[:, 0]), 1:]

    # use mapping to emilimate -1 rows
    mapping = np.loadtxt(mapping, dtype=int)
    mapping = [True if item != -1 else False for item in mapping][:emb1.shape[0]]


    # print(emb1[:, 0])
    # print('-----------------------')

    # emb1 = emb1[:emb2.shape[0], :]
    
    # delete row in emb1 if mapping is -1
    emb1 = emb1[list(compress(mapping, [True] * emb1.shape[0])), 1:]
    # print(emb1[:, 0])

    # compute cosine distance on each row pair of emb1 and emb2
    dist = 0
    for i in range(emb1.shape[0]):
        cos_dist = np.dot(emb1[i], emb2[i]) / (np.linalg.norm(emb1[i]) * np.linalg.norm(emb2[i]))
        # print(cos_dist)
        dist += abs(cos_dist)
        # dist += cos_dist
    return dist / emb1.shape[0]


def main():
    path1 = '/data3/chenyh/snap/examples/node2vec/emb/email/embedding.emb'
    prune_algo = "in_degree"
    for p in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
    # for p in ['0.2']:
        path2 = f'/data3/chenyh/snap/examples/node2vec/emb/email/{prune_algo}/{p}/embedding.emb'
        mapping = f'/data3/chenyh/sparsification/data/email/pruned/{prune_algo}/{p}/final.el.map'
        print(cos_distance(path1, path2, mapping))


if __name__ == '__main__':
    main()

