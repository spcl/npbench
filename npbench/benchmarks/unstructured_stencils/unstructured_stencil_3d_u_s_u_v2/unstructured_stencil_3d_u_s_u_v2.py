import numpy as np


def initialize(N, datatype=np.float64, seed=42):
    np.random.seed(seed)

    vals_A = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N, (N, N, N), dtype=datatype)
    vals_B = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N, (N, N, N), dtype=datatype)
    neighbors = np.random.randint(1, N, size=(N, N, 8), dtype=np.int64)
    return vals_A, vals_B, neighbors
