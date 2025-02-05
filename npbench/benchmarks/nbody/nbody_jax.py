# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

@jax.jit
def getAcc(pos, mass, G, softening):
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
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3 = jnp.where(inv_r3 > 0, inv_r3**(-1.5), inv_r3)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = jnp.hstack((ax, ay, az))

    return a

@jax.jit
def getEnergy(pos, vel, mass, G):
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
    KE = 0.5 * jnp.sum(mass * vel**2)

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = jnp.sqrt(dx**2 + dy**2 + dz**2)
    inv_r = jnp.where(inv_r > 0, 1.0 / inv_r, inv_r)

    # sum over upper triangle, to count each interaction only once
    # PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    PE = G * jnp.sum(jnp.triu(-(mass * mass.T) * inv_r, 1))

    return KE, PE

@partial(jax.jit, static_argnums=(4,))
def nbody(mass, pos, vel, N, Nt, dt, G, softening):

    # Convert to Center-of-Mass frame
    vel -= jnp.mean(mass * vel, axis=0) / jnp.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE = jnp.empty(Nt + 1, dtype=jnp.float64)
    PE = jnp.empty(Nt + 1, dtype=jnp.float64)
    ke, pe = getEnergy(pos, vel, mass, G)
    KE = KE.at[0].set(ke)
    PE = PE.at[0].set(pe)

    t = 0.0

    def loop_body(i, loop_vars):
        pos, vel, acc, KE, PE, t = loop_vars

        # (1/2) kick
        vel += acc * dt / 2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt / 2.0

        # update time
        t += dt

        # get energy of system
        ke, pe = getEnergy(pos, vel, mass, G)

        KE = KE.at[i + 1].set(ke)
        PE = PE.at[i + 1].set(pe)

        return pos, vel, acc, KE, PE, t

    # Simulation Main Loop
    pos, vel, acc, KE, PE, t = lax.fori_loop(0, Nt, loop_body, (pos, vel, acc, KE, PE, t)) 

    return KE, PE
