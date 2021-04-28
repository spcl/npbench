# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, nnz):
    from numpy.random import default_rng
    rng = default_rng(42)

    x = rng.random((N, ))

    # Randomize sparse matrix, assuming uniform sparsity across rows
    rows = np.ndarray(M + 1, dtype=np.uint32)
    cols = np.ndarray(nnz, dtype=np.uint32)
    vals = rng.random((nnz, ))
    nnz_per_row = nnz // M

    # Fill row data
    rows[0] = 0
    rows[1:M] = nnz_per_row
    rows = np.cumsum(rows, dtype=np.uint32)

    # Fill column data
    for i in range(M):
        cols[nnz_per_row*i:nnz_per_row*(i+1)] = \
            np.sort(rng.choice(N, nnz_per_row, replace=False))

    return rows, cols, vals, x
