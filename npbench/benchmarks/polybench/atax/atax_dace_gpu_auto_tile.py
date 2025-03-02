import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def _kernel(A: dc.float64[M, N], x: dc.float64[N]):

    return (A @ x) @ A

_best_config = None

def autotuner(A, x):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A, "x": x},
        dims=get_max_ndim([A, x])
    )

def kernel(A, x):
    global _best_config
    R = _best_config(A, x)
    return R
