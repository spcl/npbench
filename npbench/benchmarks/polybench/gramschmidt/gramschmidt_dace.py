import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))


@dc.program
def kernel(A: dc_float[M, N]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R
