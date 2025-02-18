import numpy as np

def initialize(N, datatype=np.float64):
    A = np.fromfunction(lambda i, j, k: (i*N*N + j*N + k*N) / N, (N, N, N), dtype=datatype)
    return A