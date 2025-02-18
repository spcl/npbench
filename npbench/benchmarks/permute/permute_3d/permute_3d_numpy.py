import numpy as np

def kernel(A, B):
    B = np.transpose(A, (2, 1, 0))
    return B