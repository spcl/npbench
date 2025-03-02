import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(L: dc.float64[N, N], x: dc.float64[N], b: dc.float64[N]):

    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

_best_config = None

def autotuner(L, x, b, N):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"L": L, "x": x, "b": b, "N": N},
        dims=get_max_ndim([L, x, b])
    )

def kernel(L, x, b, N):
    global _best_config
    _best_config(L, x, b, N)
    return b
