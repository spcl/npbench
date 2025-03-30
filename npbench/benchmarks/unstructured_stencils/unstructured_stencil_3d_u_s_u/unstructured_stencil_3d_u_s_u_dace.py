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
def kernel(
    TSTEPS: dace.int64,
    vals_A: dace.float64[N, N, N],
    vals_B: dace.float64[N, N, N],
    neighbors: dace.int64[N, N, 8],
):

    for _ in range(1, TSTEPS):
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_B[i + 1, j + 1, k + 1] = 0.2 * (
                vals_A[i + 1, j + 1, k + 1]
                + vals_A[i + 1, j + 1, k]
                + vals_A[i + 1, j + 1, k + 2]
                + vals_A[i + 1 + neighbors[i, j, 0], j + 1, k + 1 + neighbors[i, j, 4]]
                + vals_A[i + 1 + neighbors[i, j, 1], j + 1, k + 1 + neighbors[i, j, 5]]
                + vals_A[i + 1 + neighbors[i, j, 2], j + 1, k + 1 + neighbors[i, j, 6]]
                + vals_A[i + 1 + neighbors[i, j, 3], j + 1, k + 1 + neighbors[i, j, 7]]
            )
        for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
            vals_A[i + 1, j + 1, k + 1] = 0.2 * (
                vals_B[i + 1, j + 1, k + 1]
                + vals_B[i + 1, j + 1, k]
                + vals_B[i + 1, j + 1, k + 2]
                + vals_B[i + 1 + neighbors[i, j, 0], j + 1, k + 1 + neighbors[i, j, 4]]
                + vals_B[i + 1 + neighbors[i, j, 1], j + 1, k + 1 + neighbors[i, j, 5]]
                + vals_B[i + 1 + neighbors[i, j, 2], j + 1, k + 1 + neighbors[i, j, 6]]
                + vals_B[i + 1 + neighbors[i, j, 3], j + 1, k + 1 + neighbors[i, j, 7]]
            )
    return vals_A