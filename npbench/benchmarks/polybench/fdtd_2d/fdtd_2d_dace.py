import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

TMAX, NX, NY = (dc.symbol(s, dtype=dc.int64) for s in ('TMAX', 'NX', 'NY'))


@dc.program
def kernel(ex: dc_float[NX, NY], ey: dc_float[NX, NY],
           hz: dc_float[NX, NY], _fict_: dc_float[TMAX]):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])
