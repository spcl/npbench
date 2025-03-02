# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import dace as dc

NR, NM, slab_per_bc = (dc.symbol(s, dtype=dc.int64)
                       for s in ('NR', 'NM', 'slab_per_bc'))


@dc.program
def _contour_integral(Ham: dc.complex128[slab_per_bc + 1, NR, NR],
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

_best_config = None

def autotuner(Ham, int_pts, Y):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _contour_integral.to_sdfg(),
        {"Ham": Ham, "int_pts": int_pts, "Y": Y},
        dims=get_max_ndim([Ham, int_pts, Y])
    )

def contour_integral(Ham, int_pts, Y):
    global _best_config
    P0, P1 = _best_config(Ham, int_pts, Y)
    return P0, P1
