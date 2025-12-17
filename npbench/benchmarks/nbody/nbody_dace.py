# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float
"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

N, Nt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'Nt'))

# @dc.program
# def hstack(out: dc.float64[N, 3], a: dc.float64[N],
#            b: dc.float64[N], c: dc.float64[N]):
#     out[:, 0] = a
#     out[:, 1] = b
#     out[:, 2] = c


@dc.program
def getAcc(pos: dc_float[N, 3], mass: dc_float[N], G: dc_float,
           softening: dc_float):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    # dx = x.T - x
    # dy = y.T - y
    # dz = z.T - z
    # dx = np.transpose(x) - x
    # dy = np.transpose(y) - y
    # dz = np.transpose(z) - z
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    # inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
    I = inv_r3 > 0
    np.power(inv_r3, -1.5, out=inv_r3, where=I)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    # a = np.hstack((ax,ay,az))
    a = np.ndarray((N, 3), dtype=dc_float)
    # hstack(a, ax, ay, az)
    a[:, 0] = ax
    a[:, 1] = ay
    a[:, 2] = az

    return a


@dc.program
def getEnergy(pos: dc_float[N, 3], vel: dc_float[N, 3],
              mass: dc_float[N], G: dc_float):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    # KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # KE = 0.5 * np.sum( mass * vel**2 )
    KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    # dx = x.T - x
    # dy = y.T - y
    # dz = z.T - z
    # dx = np.transpose(x) - x
    # dy = np.transpose(y) - y
    # dz = np.transpose(z) - z
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    # inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]
    I = inv_r > 0
    np.divide(1.0, inv_r, out=inv_r, where=I)

    # sum over upper triangle, to count each interaction only once
    # PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    # PE = G * np.sum(np.triu(-(mass*mass.T)*inv_r,1))
    tmp = -np.multiply.outer(mass, mass) * inv_r
    PE = 0.0
    for j in range(N):
        for k in range(j + 1, N):
            PE += tmp[j, k]
    PE *= G

    return KE, PE


@dc.program
def nbody(mass: dc_float[N], pos: dc_float[N, 3], vel: dc_float[N, 3],
          dt: dc_float, G: dc_float, softening: dc_float):

    # Convert to Center-of-Mass frame
    # vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
    # vel -= np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass)
    # tmp = np.divide(np.mean(np.reshape(mass, (N, 1)) * vel, axis=0), np.mean(mass))
    np.subtract(vel,
                np.mean(np.reshape(mass,
                                   (N, 1)) * vel, axis=0) / np.mean(mass),
                out=vel)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE = np.ndarray(Nt + 1, dtype=dc_float)
    PE = np.ndarray(Nt + 1, dtype=dc_float)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)

    t = 0.0

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc[:] = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)

    return KE, PE
