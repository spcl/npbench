import numpy as np


def initialize(N, datatype=np.float64):
    vals_A = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N, (N, N, N), dtype=datatype)
    vals_B = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N, (N, N, N), dtype=datatype)
    neighbors = np.empty((N, N, N, 4), dtype=np.int64)
    neighbors[..., 0] = -1
    neighbors[..., 1] = -N
    neighbors[..., 2] = 1
    neighbors[..., 3] = N
    return vals_A, vals_B, neighbors
