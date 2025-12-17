# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float32):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i + j) % 100) / M, (M, N),
                        dtype=datatype)
    B = np.fromfunction(lambda i, j: ((N + i - j) % 100) / M, (M, N),
                        dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, :i + 1] = np.fromfunction(lambda j: ((i + j) % 100) / M,
                                       (i + 1, ),
                                       dtype=datatype)
        A[i, i + 1:] = -999

    return alpha, beta, C, A, B
