import numpy as np
import dace as dc
NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))

@dc.program
def kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ], A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):
    C[:] = alpha * A @ B + beta * C
    return (alpha, beta, C, A, B)