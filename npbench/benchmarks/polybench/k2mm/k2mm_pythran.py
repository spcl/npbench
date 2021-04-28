import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:])
def kernel(alpha, beta, A, B, C, D):

    D[:] = alpha * A @ B @ C + beta * D
