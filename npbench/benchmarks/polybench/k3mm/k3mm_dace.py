import numpy as np
import dace as dc

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64)
                      for s in ('NI', 'NJ', 'NK', 'NL', 'NM'))


@dc.program
def kernel(A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NM],
           D: dc.float64[NM, NL]):

    return A @ B @ C @ D
