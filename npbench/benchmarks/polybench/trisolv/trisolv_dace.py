import numpy as np
import dace as dc
N = dc.symbol('N', dtype=dc.int64)

@dc.program
def kernel(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N]):
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return (L, x, b)