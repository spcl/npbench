import numpy as np


def kernel(TSTEPS, vals_A, vals_B, neighbors):
    N = vals_A.shape[0]

    for _ in range(1, TSTEPS):
        vals_B[1 : N - 1, 1 : N - 1, 1 : N - 1] = 0.2 * (
            vals_A[1 : N - 1, 1 : N - 1, 1 : N - 1]
            + vals_A[1 : N - 1, 1 : N - 1, 0 : N - 2]  # k - 1
            + vals_A[1 : N - 1, 1 : N - 1, 2:N]  # k + 1
            + vals_A[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 0],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_A[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 1],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_A[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 2],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_A[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 3],
                1 : N - 1,
                1 : N - 1,
            ]
        )

        vals_A[1 : N - 1, 1 : N - 1, 1 : N - 1] = 0.2 * (
            vals_B[1 : N - 1, 1 : N - 1, 1 : N - 1]
            + vals_B[1 : N - 1, 1 : N - 1, 0 : N - 2]
            + vals_B[1 : N - 1, 1 : N - 1, 2:N]
            + vals_B[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 0],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_B[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 1],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_B[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 2],
                1 : N - 1,
                1 : N - 1,
            ]
            + vals_B[
                1 : N - 1,
                1 + neighbors[: N - 2, : N - 2, : N - 2, 3],
                1 : N - 1,
                1 : N - 1,
            ]
        )

    return vals_A
