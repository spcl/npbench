# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def _compute(array_1: dc.int64[M, N], array_2: dc.int64[M, N], a: dc.int64,
            b: dc.int64, c: dc.int64):
    r = np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c
    return r

_best_config = None

def autotuner(array_1, array_2, a, b, c):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _compute.to_sdfg(),
        {"array_1": array_1, "array_2": array_2, "a": a, "b": b, "c": c},
        dims=get_max_ndim([array_1, array_2, a, b, c])
    )

def compute(array_1, array_2, a, b, c):
    global _best_config
    r = _best_config(array_1=array_1, array_2=array_2, a=a, b=b, c=c)
    return r
