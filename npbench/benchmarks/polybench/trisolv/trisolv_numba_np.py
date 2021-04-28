import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
