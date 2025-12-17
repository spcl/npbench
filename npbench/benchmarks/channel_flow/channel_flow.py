# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(ny, nx, datatype=np.float32):
    u = np.zeros((ny, nx), dtype=datatype)
    v = np.zeros((ny, nx), dtype=datatype)
    p = np.ones((ny, nx), dtype=datatype)
    dx = datatype(2 / (nx - 1))
    dy = datatype(2 / (ny - 1))
    dt = datatype(.1 / ((nx - 1) * (ny - 1)))
    return u, v, p, dx, dy, dt
