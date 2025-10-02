import numpy as np
import dace as dc
N = dc.symbol('N', dtype=dc.int64)

@dc.program
def _kernel(x1: dc.float64[N], x2: dc.float64[N], y_1: dc.float64[N], y_2: dc.float64[N], A: dc.float64[N, N]):
    x1 += A @ y_1
    x2 += y_2 @ A
    return (x1, x2, y_1, y_2, A)
_best_config = None

def autotuner(x1, x2, y_1, y_2, A):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'x1': x1, 'x2': x2, 'y_1': y_1, 'y_2': y_2, 'A': A}, dims=get_max_ndim([x1, x2, y_1, y_2, A]))

def kernel(x1, x2, y_1, y_2, A):
    global _best_config
    _best_config(x1, x2, y_1, y_2, A)
    return x1