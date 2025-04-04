import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(M, float_n, data):
    return np.cov(np.transpose(data))
