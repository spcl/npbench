import numpy as np
import numba as nb

@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def kernel(NR, NQ, NP, A, C4):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    return (NR, NQ, NP, A, C4)