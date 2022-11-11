# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(M, N):
    from numpy.random import default_rng
    rng = default_rng(42)
    a = rng.random((M, N))
    return a
