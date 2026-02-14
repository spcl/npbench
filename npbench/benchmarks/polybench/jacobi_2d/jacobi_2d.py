# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float32):
    A = np.fromfunction(lambda i, j: i * (j + 2) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N, (N, N), dtype=datatype)

    return A, B
