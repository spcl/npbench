# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(K, M, N):
    from numpy.random import default_rng
    rng = default_rng(42)
    A = rng.random((K, M))
    B = rng.random((K, N))
    W = rng.random((K, M))
    return A, B, W
