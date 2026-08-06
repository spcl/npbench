import numpy as np


# pythran export kernel(float64[:,:])
def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
