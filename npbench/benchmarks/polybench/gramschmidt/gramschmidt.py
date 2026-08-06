# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(M, N, datatype=np.float64):
    from numpy.random import default_rng
    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    # Add a diagonal-dominance term so A is deterministically full column rank and
    # well-conditioned. A plain random matrix is only full-rank in expectation (hence the
    # former reject-sampling ``while matrix_rank(A) < N`` loop) and can still be poorly
    # conditioned, which makes the Gram-Schmidt QR numerically unstable; the added diagonal
    # gives cond(A) ~= 1.5 without the nondeterministic resampling.
    A[:N, :N] += N * np.eye(N, dtype=datatype)

    return A
