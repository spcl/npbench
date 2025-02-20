import cupy as cp

def kernel(A, B):
    B = cp.transpose(A, (2, 1, 0)).copy()
    return B