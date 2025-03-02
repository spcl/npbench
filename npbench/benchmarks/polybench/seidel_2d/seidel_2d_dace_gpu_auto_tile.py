import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(TSTEPS: dc.int64, A: dc.float64[N, N]):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                           A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0

_best_config = None

def autotuner(TSTEPS, N, A):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"TSTEPS": TSTEPS, "N": N, "A": A},
        dims=get_max_ndim([TSTEPS, A])
    )

def kernel(TSTEPS, N, A):
    global _best_config
    _best_config(TSTEPS=TSTEPS, N=N, A=A)
    return A
