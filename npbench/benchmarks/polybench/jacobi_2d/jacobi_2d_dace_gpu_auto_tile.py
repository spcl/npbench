import numpy as np
import dace as dc
from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework


N = dc.symbol("N", dtype=dc.int64)


@dc.program
def _kernel(TSTEPS: dc.int64, A: dc.float64[N, N], B: dc.float64[N, N]):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (
            A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1]
        )
        A[1:-1, 1:-1] = 0.2 * (
            B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1]
        )


_best_configg = None


def autotuner(TSTEPS, A, B, N):
    global _best_configg
    if _best_configg is not None:
        return

    __best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B, "TSTEPS": TSTEPS},
        dims=2
        )
    _best_configg = __best_config.compile()

def kernel(TSTEPS, A, B, N):
    global _best_configg
    _best_configg(TSTEPS=TSTEPS, A=A, B=B, N=N)
    return A