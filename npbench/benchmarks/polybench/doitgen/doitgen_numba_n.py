import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=False, fastmath=True)
def kernel(NR, NQ, NP, A, C4):
    for r in range(NR):
        for q in range(NQ):
            tmp = A[r, q, :] @ C4
            A[r, q, :] = tmp
    return (NR, NQ, NP, A, C4)