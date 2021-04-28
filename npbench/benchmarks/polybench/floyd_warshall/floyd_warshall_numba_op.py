import numpy as np
import numba as nb


@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def kernel(path):

    for k in range(path.shape[0]):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
