import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)