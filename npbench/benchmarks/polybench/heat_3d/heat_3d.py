# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float32):
    A = np.fromfunction(lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N),
                        dtype=datatype)
    B = np.copy(A)

    return A, B
