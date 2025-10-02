import numpy as np
import dace as dc
NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ('NI', 'NJ', 'NK'))

@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[NI, NJ], A: dc.float64[NI, NK], B: dc.float64[NK, NJ]):
    C[:] = alpha * A @ B + beta * C
    return (alpha, beta, C, A, B)
_best_config = None

def autotuner(alpha, beta, C, A, B):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'alpha': alpha, 'beta': beta, 'C': C, 'A': A, 'B': B}, dims=get_max_ndim([alpha, beta, C, A, B]))

def kernel(alpha, beta, C, A, B):
    global _best_config
    _best_config(alpha, beta, C, A, B)
    return C