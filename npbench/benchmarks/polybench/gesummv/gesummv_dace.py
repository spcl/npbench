import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
           B: dc.float64[N, N], x: dc.float64[N]):

    C = alpha * A @ x + beta * B @ x
    return C
