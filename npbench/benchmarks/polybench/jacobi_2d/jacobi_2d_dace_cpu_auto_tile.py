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


_jacobi_2d_best_config = None
tcount = None

def autotuner(TSTEPS, A, B, N):
    global _jacobi_2d_best_config
    global tcount
    if _jacobi_2d_best_config is not None:
        return

    _best_config, _tcount = DaceCPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B, "TSTEPS": 2}
    )
    tcount = _tcount
    _jacobi_2d_best_config = _best_config.compile()
    _best_config.save("jacobi_2d_best_config.sdfg")
    if tcount is None or _jacobi_2d_best_config is None:
        print("tcount or autotuned SDFG is None")
        raise  Exception("tcount or autotuned SDFG is None")
    os.environ['OMP_NUM_THREADS'] = str(tcount)
    if os.environ['OMP_NUM_THREADS'] != str(tcount):
        print("Setting OMP_NUM_THREADS failed")
        raise Exception("Setting OMP_NUM_THREADS failed")

def kernel(TSTEPS, A, B, N):
    global _jacobi_2d_best_config
    _jacobi_2d_best_config(TSTEPS=TSTEPS, A=A, B=B, N=N)
    return A