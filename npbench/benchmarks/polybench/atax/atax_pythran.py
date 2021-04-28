import numpy as np


# pythran export kernel(float64[:,:], float64[:])
def kernel(A, x):

    return (A @ x) @ A
