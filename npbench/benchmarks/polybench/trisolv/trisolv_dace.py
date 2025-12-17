import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(L: dc_float[N, N], x: dc_float[N], b: dc_float[N]):

    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
