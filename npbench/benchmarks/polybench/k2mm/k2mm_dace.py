import numpy as np
import dace as dc

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64)
                  for s in ('NI', 'NJ', 'NK', 'NL'))


@dc.program
def kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK],
           B: dc.float64[NK, NJ], C: dc.float64[NJ, NL], D: dc.float64[NI,
                                                                       NL]):

    D[:] = alpha * A @ B @ C + beta * D
