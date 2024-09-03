import dpnp as np

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""



def getAcc(pos, mass, G, softening):
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

    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    I = inv_r3 > 0
    np.power(inv_r3, -1.5, out=inv_r3, where=I)

    ax = G * np.dot(dx * inv_r3, mass)
    ay = G * np.dot(dy * inv_r3, mass)
    az = G * np.dot(dz * inv_r3, mass)

    a = np.hstack((ax, ay, az))

    return a

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
    KE = 0.5 * np.sum(np.reshape(mass, (N, 1)) * vel**2)

    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    dx = np.add.outer(-x, x)
    dy = np.add.outer(-y, y)
    dz = np.add.outer(-z, z)

    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    I = inv_r > 0
    np.divide(1.0, inv_r, out=inv_r, where=I)

    tmp = -np.multiply.outer(mass, mass) * inv_r
    PE = 0.0
    for j in range(N):
        for k in range(j + 1, N):
            PE += tmp[j, k]
    PE *= G

    return KE, PE


def nbody(mass, pos, vel, N, Nt, dt, G, softening):
    # Convert to Center-of-Mass frame
    mean_vel = np.mean(np.reshape(mass, (N, 1)) * vel, axis=0) / np.mean(mass)
    np.subtract(vel, mean_vel, out=vel)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE = np.ndarray(Nt + 1, dtype=np.float64)
    PE = np.ndarray(Nt + 1, dtype=np.float64)
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
