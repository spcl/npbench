# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import dace as dc

from npbench.infrastructure.dace_framework import dc_complex_float

NR, NM, slab_per_bc = (dc.symbol(s, dtype=dc.int64)
                       for s in ('NR', 'NM', 'slab_per_bc'))


@dc.program
def contour_integral(Ham: dc_complex_float[slab_per_bc + 1, NR, NR],
                     int_pts: dc_complex_float[32], Y: dc_complex_float[NR, NM]):
    P0 = np.zeros((NR, NM), dtype=dc_complex_float)
    P1 = np.zeros((NR, NM), dtype=dc_complex_float)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), dtype=dc_complex_float)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        # if NR == NM:
        #     X = np.linalg.inv(Tz)
        # else:
        X = np.linalg.solve(Tz, Y)
        if np.absolute(z) < 1.0:
            X[:] = -X
        P0 += X
        P1 += z * X

    return P0, P1
