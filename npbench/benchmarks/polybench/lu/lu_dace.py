import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(A: dc.float64[N, N]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
