import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(A: dc.float64[N, N], b: dc.float64[N]):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    return x, y

_best_config = None

def autotuner(A, b):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A, "b": b},
        dims=get_max_ndim([A, b])
    )

def kernel(A, b):
    global _best_config
    x, y = _best_config(A, b)
    return x, y
