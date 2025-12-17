# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def go_fast(a: dc_float[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace
