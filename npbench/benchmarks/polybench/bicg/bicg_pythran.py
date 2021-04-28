import numpy as np


# pythran export kernel(float64[:,:], float64[:], float64[:])
def kernel(A, p, r):

    return r @ A, A @ p
