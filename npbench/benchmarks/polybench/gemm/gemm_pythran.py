import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:], float64[:,:])
def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C
