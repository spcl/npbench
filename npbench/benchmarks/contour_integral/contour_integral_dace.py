# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import dace as dc

NR, NM, slab_per_bc = (dc.symbol(s, dtype=dc.int64)
                       for s in ('NR', 'NM', 'slab_per_bc'))


@dc.program
def contour_integral(Ham: dc.complex128[slab_per_bc + 1, NR, NR],
                     int_pts: dc.complex128[32], Y: dc.complex128[NR, NM]):
    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    for idx in range(32):
        z = int_pts[idx]
        Tz = np.zeros((NR, NR), dtype=np.complex128)
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
