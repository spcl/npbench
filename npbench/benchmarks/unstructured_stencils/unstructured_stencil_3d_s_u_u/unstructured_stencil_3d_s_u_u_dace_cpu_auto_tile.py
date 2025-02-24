import copy
import typing
import dace
import numpy as np

from dace.sdfg.utils import inline_sdfgs
import dace.transformation
from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG

from npbench.infrastructure import DaceCPUAutoTileFramework

N = dace.symbol("N")


@dace.program
def _kernel(
    TSTEPS: dace.int64,
    vals_A: dace.float64[N, N, N],
    vals_B: dace.float64[N, N, N],
    neighbors: dace.int64[N, N, N, 8],
):

    for _ in range(1, TSTEPS):
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_B[i + 1, j + 1, k + 1] = 0.2 * (
                vals_A[i + 1, j + 1, k + 1]
                + vals_A[i + 1, j + 1, k]
                + vals_A[i + 1, j + 1, k + 2]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 0], k + 1 + neighbors[i, j, k, 4]]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 1], k + 1 + neighbors[i, j, k, 5]]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 2], k + 1 + neighbors[i, j, k, 6]]
                + vals_A[i + 1, j + 1 + neighbors[i, j, k, 3], k + 1 + neighbors[i, j, k, 7]]
            )
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_A[i + 1, j + 1, k + 1] = 0.2 * (
                vals_B[i + 1, j + 1, k + 1]
                + vals_B[i + 1, j + 1, k]
                + vals_B[i + 1, j + 1, k + 2]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 0], k + 1 + neighbors[i, j, k, 4]]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 1], k + 1 + neighbors[i, j, k, 5]]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 2], k + 1 + neighbors[i, j, k, 6]]
                + vals_B[i + 1, j + 1 + neighbors[i, j, k, 3], k + 1 + neighbors[i, j, k, 7]]
            )
    return vals_A


_semi_structured_3d_s_u_s_best_config = None


def autotuner(TSTEPS, vals_A, vals_B, neighbors, N):
    global _semi_structured_3d_s_u_s_best_config
    if _semi_structured_3d_s_u_s_best_config is not None:
        return

    _sdfg = _kernel.to_sdfg()

    _best_config = DaceCPUAutoTileFramework.autotune(
        _sdfg,
        {
            "N": N,
            "vals_A": vals_A,
            "vals_B": vals_B,
            "neighbors": neighbors,
            "TSTEPS": TSTEPS,
        },
    )
    _semi_structured_3d_s_u_s_best_config = _best_config.compile()


def kernel(TSTEPS, vals_A, vals_B, neighbors, N):
    global _semi_structured_3d_s_u_s_best_config
    _semi_structured_3d_s_u_s_best_config(
        TSTEPS=TSTEPS, vals_A=vals_A, vals_B=vals_B, neighbors=neighbors, N=N
    )
    return vals_A
