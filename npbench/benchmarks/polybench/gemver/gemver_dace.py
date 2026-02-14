import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(alpha: dc_float, beta: dc_float, A: dc_float[N, N],
           u1: dc_float[N], v1: dc_float[N], u2: dc_float[N],
           v2: dc_float[N], w: dc_float[N], x: dc_float[N],
           y: dc_float[N], z: dc_float[N]):

    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
