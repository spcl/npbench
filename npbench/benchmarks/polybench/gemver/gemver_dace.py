import numpy as np
import dace as dc
M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N], u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
    return (alpha, beta, A, u1, v1, u2, v2, w, x, y, z)