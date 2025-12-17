import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def flip(A: dc_float[M]):
    B = np.ndarray((M, ), dtype=dc_float)
    for i in dc.map[0:M]:
        B[i] = A[M - 1 - i]
    return B

@dc.program
def kernel(r: dc_float[N]):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * flip(y[:k])
        y[k] = alpha

    return y
