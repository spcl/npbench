import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))


@dc.program
def kernel(alpha: dc_float, beta: dc_float, C: dc_float[NI, NJ],
           A: dc_float[NI, NK], B: dc_float[NK, NJ]):

    C[:] = alpha * A @ B + beta * C
