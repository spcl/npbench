import numpy as np
import dace as dc
M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N], A: dc.float64[M, M], B: dc.float64[M, N]):
    temp2 = np.empty((N,), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
    return (alpha, beta, C, A, B)