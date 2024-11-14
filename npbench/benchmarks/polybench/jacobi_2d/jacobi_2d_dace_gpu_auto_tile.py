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


_jacobi_2d_triton_best_config = None


def autotuner(TSTEPS, A, B, N):
    global _jacobi_2d_triton_best_config
    if _jacobi_2d_triton_best_config is not None:
        return
    _jacobi_2d_triton_best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N}
        ).compile()


def kernel(TSTEPS, A, B, N):
    global _jacobi_2d_triton_best_config
    _jacobi_2d_triton_best_config(TSTEPS, A, B, N)