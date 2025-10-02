import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(path):
    for k in range(path.shape[0]):
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
    return path