# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _go_fast(a: dc.float64[N, N]):
    trace = 0.0
    for i in range(N):
        trace += np.tanh(a[i, i])
    r = a + trace
    return r

_best_config = None

def autotuner(a, N):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _go_fast.to_sdfg(),
        {"a": a, "N":N},
        dims=get_max_ndim([a])
    )

def go_fast(a):
    global _best_config
    r = _best_config(a, N)
    return r
