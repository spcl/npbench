# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.


def initialize(N):
    from numpy.random import default_rng
    rng = default_rng(42)
    t0, p0, t1, p1 = rng.random((N, )), rng.random((N, )), rng.random(
        (N, )), rng.random((N, ))
    return t0, p0, t1, p1
