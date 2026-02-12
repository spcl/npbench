# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float32):
    A = np.fromfunction(lambda i: (i + 2) / N, (N, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N, ), dtype=datatype)

    return A, B
