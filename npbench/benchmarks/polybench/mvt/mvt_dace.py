import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(x1: dc_float[N], x2: dc_float[N], y_1: dc_float[N],
           y_2: dc_float[N], A: dc_float[N, N]):

    x1 += A @ y_1
    x2 += y_2 @ A
