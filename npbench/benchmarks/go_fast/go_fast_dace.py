# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def go_fast(a: dc.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    return a + trace
