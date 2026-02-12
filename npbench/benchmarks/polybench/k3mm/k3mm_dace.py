import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64)
                      for s in ('NI', 'NJ', 'NK', 'NL', 'NM'))


@dc.program
def kernel(A: dc_float[NI, NK], B: dc_float[NK, NJ], C: dc_float[NJ, NM],
           D: dc_float[NM, NL]):

    return A @ B @ C @ D
