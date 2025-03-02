import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N],
           B: dc.float64[N, N], x: dc.float64[N]):

    C = alpha * A @ x + beta * B @ x
    return C

_best_config = None

def autotuner(alpha, beta, A, B, x):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"alpha": alpha, "beta": beta, "A": A, "B": B, "x": x},
        dims=get_max_ndim([alpha, beta, A, B, x])
    )

def kernel(alpha, beta, A, B, x):
    global _best_config
    C = _best_config(alpha, beta, A, B, x)
    return C
