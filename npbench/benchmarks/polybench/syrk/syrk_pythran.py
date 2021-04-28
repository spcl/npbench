import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:])
def kernel(alpha, beta, C, A):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
