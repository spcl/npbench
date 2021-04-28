# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, H, W):
    from numpy.random import default_rng
    rng = default_rng(42)

    H_conv1 = H - 4
    W_conv1 = W - 4
    H_pool1 = H_conv1 // 2
    W_pool1 = W_conv1 // 2
    H_conv2 = H_pool1 - 4
    W_conv2 = W_pool1 - 4
    H_pool2 = H_conv2 // 2
    W_pool2 = W_conv2 // 2
    C_before_fc1 = 16 * H_pool2 * W_pool2

    # NHWC data layout
    input = rng.random((N, H, W, 1), dtype=np.float32)
    # Weights
    conv1 = rng.random((5, 5, 1, 6), dtype=np.float32)
    conv1bias = rng.random((6, ), dtype=np.float32)
    conv2 = rng.random((5, 5, 6, 16), dtype=np.float32)
    conv2bias = rng.random((16, ), dtype=np.float32)
    fc1w = rng.random((C_before_fc1, 120), dtype=np.float32)
    fc1b = rng.random((120, ), dtype=np.float32)
    fc2w = rng.random((120, 84), dtype=np.float32)
    fc2b = rng.random((84, ), dtype=np.float32)
    fc3w = rng.random((84, 10), dtype=np.float32)
    fc3b = rng.random((10, ), dtype=np.float32)

    return (input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
            fc3w, fc3b, C_before_fc1)
