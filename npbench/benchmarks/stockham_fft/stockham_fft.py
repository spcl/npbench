# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(R, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    N = R**K
    X = rng_complex((N, ), rng)
    Y = np.zeros_like(X, dtype=np.complex128)

    return N, X, Y
