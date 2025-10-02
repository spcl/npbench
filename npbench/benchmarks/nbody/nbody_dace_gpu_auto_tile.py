import numpy as np
import dace as dc
"\nCreate Your Own N-body Simulation (With Python)\nPhilip Mocz (2020) Princeton Univeristy, @PMocz\nSimulate orbits of stars interacting due to gravity\nCode calculates pairwise forces according to Newton's Law of Gravity\n"
N, Nt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'Nt'))

@dc.program
def getAcc(pos: dc.float64[N, 3], mass: dc.float64[N], G: dc.float64, softening: dc.float64):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)
    inv_r3 = dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2
    I = inv_r3 > 0
    np.power(inv_r3, -1.5, out=inv_r3, where=I)
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    a = np.ndarray((N, 3), dtype=np.float64)
    a[:, 0] = ax
    a[:, 1] = ay
    a[:, 2] = az
    return a

@dc.program
def getEnergy(pos: dc.float64[N, 3], vel: dc.float64[N, 3], mass: dc.float64[N], G: dc.float64):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel ** 2)
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)
    inv_r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    I = inv_r > 0
    np.divide(1.0, inv_r, out=inv_r, where=I)
    tmp = -np.multiply.outer(mass, mass) * inv_r
    PE = 0.0
    for j in range(N):
        for k in range(j + 1, N):
            PE += tmp[j, k]
    PE *= G
    return (KE, PE)

@dc.program
def _nbody(mass: dc.float64[N], pos: dc.float64[N, 3], vel: dc.float64[N, 3], dt: dc.float64, G: dc.float64, softening: dc.float64, KE: dc.float64[N + 1], PE: dc.float64[N + 1]):
    np.subtract(vel, np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass), out=vel)
    acc = getAcc(pos, mass, G, softening)
    KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    t = 0.0
    for i in range(Nt):
        vel += acc * dt / 2.0
        pos += vel * dt
        acc[:] = getAcc(pos, mass, G, softening)
        vel += acc * dt / 2.0
        t += dt
        KE[i + 1], PE[i + 1] = getEnergy(pos, vel, mass, G)
    return (mass, pos, vel, dt, G, softening, KE, PE)
_best_config = None

def autotuner(mass, pos, vel, dt, G, softening, KE, PE, N, Nt, tEnd):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_nbody.to_sdfg(), {'mass': mass, 'pos': pos, 'vel': vel, 'dt': dt, 'G': G, 'softening': softening, 'KE': KE, 'PE': PE, 'N': N, 'Nt': Nt, 'tEnd': tEnd}, dims=get_max_ndim([mass, pos, vel, dt, G, softening, KE, PE]))

def nbody(mass, pos, vel, dt, G, softening, KE, PE, N, Nt, tEnd):
    global _best_config
    _best_config(mass, pos, vel, dt, G, softening, KE, PE, N, Nt, tEnd)
    return (KE, PE)