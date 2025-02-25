import copy
import typing
import dace
import numpy as np

from dace.sdfg.utils import inline_sdfgs
import dace.transformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG

from npbench.infrastructure import DaceGPUAutoTileFramework

N = dace.symbol("N")


@dace.program
def _kernel(
    TSTEPS: dace.int64,
    vals_A: dace.float64[N, N, N],
    vals_B: dace.float64[N, N, N],
    neighbors: dace.int64[N, N, N, 4],
):

    for _ in range(1, TSTEPS):
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_B[i + 1, j + 1, k + 1] = 0.2 * (
                vals_A[i + 1, j + 1, k + 1]
                + vals_A[i + 1, j + 1, k]
                + vals_A[i + 1, j + 1, k + 2]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 0], k + 1]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 1], k + 1]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 2], k + 1]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 3], k + 1]
            )
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_A[i + 1, j + 1, k + 1] = 0.2 * (
                vals_B[i + 1, j + 1, k + 1]
                + vals_B[i + 1, j + 1, k]
                + vals_B[i + 1, j + 1, k + 2]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 0], k + 1]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 1], k + 1]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 2], k + 1]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 3], k + 1]
            )
    return vals_A


_best_config = None


def autotuner(TSTEPS, vals_A, vals_B, neighbors, N):
    global _best_config
    if _best_config is not None:
        return

    _sdfg = _kernel.to_sdfg()

    _best_config = DaceGPUAutoTileFramework.autotune(
        _sdfg,
        {
            "N": N,
            "vals_A": vals_A,
            "vals_B": vals_B,
            "neighbors": neighbors,
            "TSTEPS": TSTEPS,
        },
        dims=3,
    )


def kernel(TSTEPS, vals_A, vals_B, neighbors, N):
    global _best_config
    _best_config(
        TSTEPS=TSTEPS, vals_A=vals_A, vals_B=vals_B, neighbors=neighbors, N=N
    )
    return vals_A
