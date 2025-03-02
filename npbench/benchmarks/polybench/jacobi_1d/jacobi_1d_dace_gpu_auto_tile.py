import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def _kernel(TSTEPS: dc.int64, A: dc.float64[N], B: dc.float64[N]):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])

_best_config = None

def autotuner(TSTEPS, A, B, N):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"TSTEPS": TSTEPS, "A": A, "B": B, "N": N},
        dims=get_max_ndim([TSTEPS, A, B])
    )

def kernel(TSTEPS, A, B, N):
    global _best_config
    _best_config(TSTEPS=TSTEPS, A=A, B=B, N=N)
    return A
