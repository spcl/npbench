import numpy as np
import dace as dc
TMAX, NX, NY = (dc.symbol(s, dtype=dc.int64) for s in ('TMAX', 'NX', 'NY'))

@dc.program
def kernel(ex: dc.float64[NX, NY], ey: dc.float64[NX, NY], hz: dc.float64[NX, NY], _fict_: dc.float64[TMAX]):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])
    return (ex, ey, hz, _fict_)