# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import numpy as np
import dace as dc

N, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'npt'))


@dc.program
def _azimint_naive(data: dc.float64[N], radius: dc.float64[N]):
    # rmax = radius.max()
    rmax = np.amax(radius)
    res = np.zeros((npt, ), dtype=np.float64)  # Fix in np.full
    for i in range(npt):
        # for i in dc.map[0:npt]:  # Optimization
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        # values_r12 = data[mask_r12]
        # res[i] = np.mean(values_r12)
        on_values = 0
        tmp = np.float64(0)
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res

_best_config = None

def autotuner(data, radius):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _azimint_naive.to_sdfg(),
        {"data": data, "radius": radius},
        dims=get_max_ndim([data, radius])
    )

def azimint_naive(data, radius):
    global _best_config
    res = _best_config(data, radius)
    return res
