import numpy as np
import dace as dc
M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))

@dc.program
def _kernel(alpha: dc.float64, A: dc.float64[M, M], B: dc.float64[M, N]):
    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
    return (alpha, A, B)
_best_config = None

def autotuner(alpha, A, B):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'alpha': alpha, 'A': A, 'B': B}, dims=get_max_ndim([alpha, A, B]))

def kernel(alpha, A, B):
    global _best_config
    _best_config(alpha, A, B)
    return B