# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    x, y = rng.random((N, )), rng.random((N, ))
    return x, y
