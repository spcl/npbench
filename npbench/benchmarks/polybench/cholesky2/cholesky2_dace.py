import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

N = dc.symbol('N', dtype=dc.int64)
k = dc.symbol('k', dtype=dc.int64)


@dc.program
def triu(A: dc_float[N, N], k: dc.int64):
    B = np.zeros_like(A)
    for i in dc.map[0:N]:
        for j in dc.map[i + k:N]:
            B[i, j] = A[i, j]
    return B


@dc.program
def kernel(A: dc_float[N, N]):
    A[:] = np.linalg.cholesky(A) + triu(A, k=1)
