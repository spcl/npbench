import numpy as np

def kernel(TSTEPS, vals_A, vals_B, neighbors):
    N = vals_A.shape[0]
    i, j, k = np.ogrid[1:N-1, 1:N-1, 1:N-1]  # Create index grids

    for _ in range(1, TSTEPS):
        vals_B[i, j, k] = 0.2 * (
            vals_A[i, j, k]
            + vals_A[i, j, k - 1]
            + vals_A[i, j, k + 1]
            + vals_A[i, j + neighbors[i, j, k, 0], k + neighbors[i, j, k, 4]]
            + vals_A[i, j + neighbors[i, j, k, 1], k + neighbors[i, j, k, 5]]
            + vals_A[i, j + neighbors[i, j, k, 2], k + neighbors[i, j, k, 6]]
            + vals_A[i, j + neighbors[i, j, k, 3], k + neighbors[i, j, k, 7]]
        )

        vals_A[i, j, k] = 0.2 * (
            vals_B[i, j, k]
            + vals_B[i, j, k - 1]
            + vals_B[i, j, k + 1]
            + vals_B[i, j + neighbors[i, j, k, 0], k + neighbors[i, j, k, 4]]
            + vals_B[i, j + neighbors[i, j, k, 1], k + neighbors[i, j, k, 5]]
            + vals_B[i, j + neighbors[i, j, k, 2], k + neighbors[i, j, k, 6]]
            + vals_B[i, j + neighbors[i, j, k, 3], k + neighbors[i, j, k, 7]]
        )

    return vals_A
