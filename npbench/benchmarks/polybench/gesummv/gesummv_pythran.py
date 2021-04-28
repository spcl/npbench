import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:], float64[:])
def kernel(alpha, beta, A, B, x):

    return alpha * A @ x + beta * B @ x
