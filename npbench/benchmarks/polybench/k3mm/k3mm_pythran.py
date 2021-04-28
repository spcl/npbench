import numpy as np


# pythran export kernel(float64[:,:], float64[:,:], float64[:,:], float64[:,:])
def kernel(A, B, C, D):

    return A @ B @ C @ D
