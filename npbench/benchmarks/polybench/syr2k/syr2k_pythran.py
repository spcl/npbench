import numpy as np


# pythran export kernel(float64, float64, float64[:,:], float64[:,:], float64[:,:])
def kernel(alpha, beta, C, A, B):

    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])
