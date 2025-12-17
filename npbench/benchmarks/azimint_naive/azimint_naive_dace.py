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
from npbench.infrastructure.dace_framework import dc_float

N, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'npt'))


@dc.program
def azimint_naive(data: dc_float[N], radius: dc_float[N]):
    # rmax = radius.max()
    rmax = np.amax(radius)
    res = np.zeros((npt, ), dtype=dc_float)  # Fix in np.full
    for i in range(npt):
        # for i in dc.map[0:npt]:  # Optimization
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        # values_r12 = data[mask_r12]
        # res[i] = np.mean(values_r12)
        on_values = 0
        tmp = dc_float(0)
        for j in dc.map[0:N]:
            if mask_r12[j]:
                tmp += data[j]
                on_values += 1
        res[i] = tmp / on_values
    return res
