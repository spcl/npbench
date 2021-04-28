import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(A: dc.float64[N, M], p: dc.float64[M], r: dc.float64[N]):

    return r @ A, A @ p
