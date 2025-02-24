import numpy as np


def initialize(N, datatype=np.float64):
    vals_A = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N, (N, N, N), dtype=datatype)
    vals_B = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N, (N, N, N), dtype=datatype)
    neighbors = np.empty((N, N, N, 8), dtype=np.int64)
    neighbors[..., 0] = -1
    neighbors[..., 1] = -1
    neighbors[..., 2] = 1
    neighbors[..., 3] = 1
    neighbors[..., 4] = -1
    neighbors[..., 5] = -1
    neighbors[..., 6] = 1
    neighbors[..., 7] = 1
    return vals_A, vals_B, neighbors
