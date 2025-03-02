import numpy as np
import dace as dc

M, N, S = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'S'))


@dc.program
def _kernel(A: dc.float64[M, N]):

    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R

_best_config = None

def autotuner(A):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A},
        dims=get_max_ndim([A])
    )

def kernel(A):
    global _best_config
    Q, R = _best_config(A)
    return Q, R
