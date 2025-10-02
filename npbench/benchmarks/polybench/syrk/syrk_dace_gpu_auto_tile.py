import numpy as np
import dace as dc
M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[N, N], A: dc.float64[N, M]):
    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
    return (alpha, beta, C, A)
_best_config = None

def autotuner(alpha, beta, C, A):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'alpha': alpha, 'beta': beta, 'C': C, 'A': A}, dims=get_max_ndim([alpha, beta, C, A]))

def kernel(alpha, beta, C, A):
    global _best_config
    _best_config(alpha, beta, C, A)
    return C