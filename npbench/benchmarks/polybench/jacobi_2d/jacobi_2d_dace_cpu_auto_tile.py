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


_best_configg = None
tcount = None

def autotuner(TSTEPS, A, B, N):
    global _best_configg
    global tcount
    if _best_configg is not None:
        return

    __best_config, _tcount = DaceCPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B, "TSTEPS": TSTEPS},
        dims=2
    )
    tcount = _tcount
    _best_config = __best_config.compile()
    if tcount is None or _best_configg is None:
        print("tcount or autotuned SDFG is None")
        raise  Exception("tcount or autotuned SDFG is None")
    os.environ['OMP_NUM_THREADS'] = str(tcount)
    if os.environ['OMP_NUM_THREADS'] != str(tcount):
        print("Setting OMP_NUM_THREADS failed")
        raise Exception("Setting OMP_NUM_THREADS failed")

def kernel(TSTEPS, A, B, N):
    global _best_configg
    global tcount
    assert(os.environ['OMP_NUM_THREADS'] == str(tcount))
    _best_configg(TSTEPS=TSTEPS, A=A, B=B, N=N)
    return A