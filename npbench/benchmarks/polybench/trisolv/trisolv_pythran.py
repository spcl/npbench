import numpy as np


# pythran export kernel(float64[:,:], float64[:], float64[:])
def kernel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
