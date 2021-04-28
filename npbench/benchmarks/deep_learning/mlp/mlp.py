# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng
    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0], ), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1], ), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2], ), dtype=np.float32)

    return input, w1, b1, w2, b2, w3, b3
