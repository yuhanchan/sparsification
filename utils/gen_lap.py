"""
This script convert un-weighted, no-heading edge-list input
to weighted, heading laplacian edge-list output.
"""

from scipy.sparse import csgraph, csr_matrix
import sys
import numpy as np

def main(argv):
    if len(argv) != 2:
        print("Usage: python3 gen_lap.py <input_file>")
        sys.exit(1)

    input_file = argv[1]
    pair = np.loadtxt(input_file, dtype=np.int32, usecols=(0, 1))
    val = np.ones(pair.shape[0], dtype=np.int32)
    # convert to sparse matrix
    A = csr_matrix((val, (pair[:, 0], pair[:, 1])), shape=(pair.max() + 1, pair.max() + 1))
    # print(A.toarray())

    # convert to weighted laplacian matrix
    A = csgraph.laplacian(A, normed=False)
    # print(A.toarray())
    A = A.tocsc()

    # save to file
    np.savetxt(input_file + ".lap.col_ptr", A.indptr, fmt="%d")
    np.savetxt(input_file + ".lap.row_ind", A.indices, fmt="%d")
    np.savetxt(input_file + ".lap.val", A.data, fmt="%d")

if __name__ == "__main__":
    main(sys.argv)
