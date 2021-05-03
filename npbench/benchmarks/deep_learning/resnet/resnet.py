# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, W, H, C1, C2):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float32)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float32)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float32)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float32)
    return (input, conv1, conv2, conv3)
