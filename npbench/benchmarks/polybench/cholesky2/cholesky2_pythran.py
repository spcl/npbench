import numpy as np

def kernel2(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
    return A