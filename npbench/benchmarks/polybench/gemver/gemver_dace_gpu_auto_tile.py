import numpy as np
import dace as dc
M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[N, N], u1: dc.float64[N], v1: dc.float64[N], u2: dc.float64[N], v2: dc.float64[N], w: dc.float64[N], x: dc.float64[N], y: dc.float64[N], z: dc.float64[N]):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
    return (alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
_best_config = None

def autotuner(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'alpha': alpha, 'beta': beta, 'A': A, 'u1': u1, 'v1': v1, 'u2': u2, 'v2': v2, 'w': w, 'x': x, 'y': y, 'z': z}, dims=get_max_ndim([alpha, beta, A, u1, v1, u2, v2, w, x, y, z]))

def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    global _best_config
    _best_config(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
    return A