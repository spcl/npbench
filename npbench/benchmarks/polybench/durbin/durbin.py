# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, datatype=np.float32):
    r = np.fromfunction(lambda i: N + 1 - i, (N, ), dtype=datatype)
    return r
