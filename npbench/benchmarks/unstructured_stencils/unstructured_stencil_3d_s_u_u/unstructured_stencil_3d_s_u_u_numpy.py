import numpy as np

def kernel(TSTEPS, vals_A, vals_B, neighbors):
    N = vals_A.shape[0]
    print(vals_A.shape)
    print(vals_B.shape)
    print(neighbors.shape)
    print(neighbors[:N-2, :N-2, :N-2, 0].shape)
    print(vals_A[1:N-1, 1 + neighbors[:N-2, :N-2, :N-2, 0], 1 + neighbors[:N-2, :N-2, :N-2, 4]].shape)
    i, j, k = np.ogrid[1:N-1, 1:N-1, 1:N-1]  # Create index grids
    vals_B[i, j, k] = 0.2 * (
        vals_A[i, j, k]
        + vals_A[i, j, k - 1]
        + vals_A[i, j, k + 1]
        + vals_A[i, j + neighbors[:-2, :-2, :-2, 0], k + neighbors[:-2, :-2, :-2, 4]]
        + vals_A[i, j + neighbors[:-2, :-2, :-2, 1], k + neighbors[:-2, :-2, :-2, 5]]
        + vals_A[i, j + neighbors[:-2, :-2, :-2, 2], k + neighbors[:-2, :-2, :-2, 6]]
        + vals_A[i, j + neighbors[:-2, :-2, :-2, 3], k + neighbors[:-2, :-2, :-2, 7]]
    )

    vals_A[i, j, k] = 0.2 * (
        vals_B[i, j, k]
        + vals_B[i, j, k - 1]
        + vals_B[i, j, k + 1]
        + vals_B[i, j + neighbors[:-2, :-2, :-2, 0], k + neighbors[:-2, :-2, :-2, 4]]
        + vals_B[i, j + neighbors[:-2, :-2, :-2, 1], k + neighbors[:-2, :-2, :-2, 5]]
        + vals_B[i, j + neighbors[:-2, :-2, :-2, 2], k + neighbors[:-2, :-2, :-2, 6]]
        + vals_B[i, j + neighbors[:-2, :-2, :-2, 3], k + neighbors[:-2, :-2, :-2, 7]]
    )

    return vals_A
