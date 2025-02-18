import cupy as cp

def kernel(A):
    A = cp.transpose(A, (2, 1, 0))
    return A
