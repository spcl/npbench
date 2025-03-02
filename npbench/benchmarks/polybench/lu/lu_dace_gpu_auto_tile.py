import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(A: dc.float64[N, N]):

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]

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
    _best_config(A)
    return A
