import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True, fastmath=True)
def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D
    return (alpha, beta, A, B, C, D)