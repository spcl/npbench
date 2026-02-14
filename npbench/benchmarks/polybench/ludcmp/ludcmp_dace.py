import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(A: dc_float[N, N], b: dc_float[N]):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x, y
