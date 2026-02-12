import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(A: dc_float[N, N]):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


# def kernel2(A):
#     A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
