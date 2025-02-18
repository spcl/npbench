# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np


def initialize(N, tEnd, dt):
    from numpy.random import default_rng
    rng = default_rng(42)
    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    pos = rng.random((N, 3))  # randomly selected positions and velocities
    vel = rng.random((N, 3))
    Nt = int(np.ceil(tEnd / dt))
    KE = np.zeros(Nt + 1, dtype=np.float64)
    PE = np.zeros(Nt + 1, dtype=np.float64)
    return mass, pos, vel, Nt, KE, PE
