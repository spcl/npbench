# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def rng_complex(shape, rng, datatype):
    return (rng.random(shape, dtype=datatype) + rng.random(shape, dtype=datatype) * 1j)


def initialize(R, K, datatype=np.float32):
    from numpy.random import default_rng
    rng = default_rng(42)

    N = R**K
    X = rng_complex((N,), rng, datatype)
    Y = np.zeros_like(X, dtype=X.dtype)

    return N, X, Y
