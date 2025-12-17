import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def kernel(A: dc_float[N, M], p: dc_float[M], r: dc_float[N]):

    return r @ A, A @ p
