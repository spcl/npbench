import numpy as np
import dace as dc
M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))

@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, C: dc.float64[M, N], A: dc.float64[M, M], B: dc.float64[M, N]):
    temp2 = np.empty((N,), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
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