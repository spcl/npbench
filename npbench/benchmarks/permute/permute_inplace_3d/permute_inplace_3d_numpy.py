import numpy as np

def kernel(A):
    A = np.transpose(A, (2, 1, 0))
    return A
