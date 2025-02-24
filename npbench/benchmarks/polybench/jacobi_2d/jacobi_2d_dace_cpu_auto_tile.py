import os
import numpy as np
import dace as dc
from npbench.infrastructure.dace_cpu_auto_tile_framework import DaceCPUAutoTileFramework


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


_best_config = None

def autotuner(TSTEPS, A, B, N):
    global _best_config
    if _best_config is not None:
        return

    _best_config = DaceCPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B, "TSTEPS": TSTEPS},
        dims=2
    )

def kernel(TSTEPS, A, B, N):
    global _best_config
    _best_config(TSTEPS=TSTEPS, A=A, B=B, N=N)
    return A