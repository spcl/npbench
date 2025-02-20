import numpy as np

def kernel(A, B):
    B = np.transpose(A, (2, 1, 0)).copy()
    assert not np.shares_memory(A, B)
    return B