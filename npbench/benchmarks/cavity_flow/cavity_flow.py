# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(ny, nx):
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = .1 / nx
    return u, v, p, dx, dy, dt
