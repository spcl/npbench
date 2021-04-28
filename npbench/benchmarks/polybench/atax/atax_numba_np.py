import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(A, x):

    return (A @ x) @ A
