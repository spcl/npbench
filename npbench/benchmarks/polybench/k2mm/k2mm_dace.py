import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64)
                  for s in ('NI', 'NJ', 'NK', 'NL'))


@dc.program
def kernel(alpha: dc_float, beta: dc_float, A: dc_float[NI, NK],
           B: dc_float[NK, NJ], C: dc_float[NJ, NL], D: dc_float[NI, NL]):

    D[:] = alpha * A @ B @ C + beta * D
