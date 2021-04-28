# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def compute(array_1: dc.int64[M, N], array_2: dc.int64[M, N], a: dc.int64,
            b: dc.int64, c: dc.int64):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
