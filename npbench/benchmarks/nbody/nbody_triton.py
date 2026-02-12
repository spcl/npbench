import torch
import triton
import triton.language as tl
import itertools
from triton.language.extra import libdevice

def get_configs():
    return [
        triton.Config({"BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k}, num_warps=num_warps)
        for n, k, num_warps in itertools.product(
            [8, 16], [8, 16, 32, 64, 128, 256], [1, 2, 4, 8, 16]
        )
    ]

@triton.autotune(configs=get_configs(), key=["N"], cache_results=True)
@triton.jit
def _get_acc(pos, mass, G, softening, acc, N, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations

    Below the numpy code for reference:
    """
    # # positions r = [x,y,z] for all particles
    # x = pos[:, 0:1]
    # y = pos[:, 1:2]
    # z = pos[:, 2:3]

    # # matrix that stores all pairwise particle separations: r_j - r_i
    # dx = x.T - x
    # dy = y.T - y
    # dz = z.T - z

    # # matrix that stores 1/r^3 for all particle pairwise particle separations
    # inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    # inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    # ax = G * (dx * inv_r3) @ mass
    # ay = G * (dy * inv_r3) @ mass
    # az = G * (dz * inv_r3) @ mass

    # # pack together the acceleration components
    # a = np.hstack((ax, ay, az))
    # ---------------------------------------------------------------#

    pid = tl.program_id(0)

    offs_i = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_i = offs_i < N

    xi = tl.load(pos + offs_i * 3 + 0, mask=mask_i, other=0.0)
    yi = tl.load(pos + offs_i * 3 + 1, mask=mask_i, other=0.0)
    zi = tl.load(pos + offs_i * 3 + 2, mask=mask_i, other=0.0)

    ax = tl.zeros([BLOCK_SIZE_N], dtype=DTYPE)
    ay = tl.zeros([BLOCK_SIZE_N], dtype=DTYPE)
    az = tl.zeros([BLOCK_SIZE_N], dtype=DTYPE)

    for j_start in range(0, N, BLOCK_SIZE_K):
        offs_j = j_start + tl.arange(0, BLOCK_SIZE_K)
        mask_j = offs_j < N

        xj = tl.load(pos + offs_j * 3 + 0, mask=mask_j, other=0.0)
        yj = tl.load(pos + offs_j * 3 + 1, mask=mask_j, other=0.0)
        zj = tl.load(pos + offs_j * 3 + 2, mask=mask_j, other=0.0)

        mj = tl.load(mass + offs_j, mask=mask_j, other=0.0)

        dx = xj[None, :] - xi[:, None]
        dy = yj[None, :] - yi[:, None]
        dz = zj[None, :] - zi[:, None]

        # NumPy: inv_r3 = (dx**2 + dy**2 + dz**2 + soft**2); inv_r3[>0] = inv_r3[>0]**(-1.5)
        r2 = dx * dx + dy * dy + dz * dz + softening * softening

        mask_ij = (offs_i[:, None] < N) & (offs_j[None, :] < N)

        inv_r3 = tl.zeros_like(r2)
        valid = mask_ij & (r2 > 0)
        inv_r3 = tl.where(valid, libdevice.pow(r2, -1.5), 0.0)

        mj_2d = mj[None, :]   # [1, BLOCK_SIZE_K]
        factor = G * inv_r3 * mj_2d

        ax += tl.sum(dx * factor, axis=1)
        ay += tl.sum(dy * factor, axis=1)
        az += tl.sum(dz * factor, axis=1)

    tl.store(acc + offs_i * 3 + 0, ax, mask=mask_i)
    tl.store(acc + offs_i * 3 + 1, ay, mask=mask_i)
    tl.store(acc + offs_i * 3 + 2, az, mask=mask_i)

@triton.autotune(configs=get_configs(), key=["N"], cache_results=True)
@triton.jit
def _get_energy(pos, mass, G, pe, N, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system

    Below the numpy code for reference:
    """
    # # Kinetic Energy:
    # # KE = 0.5 * np.sum(np.sum( mass * vel**2 ))
    # KE = 0.5 * np.sum(mass * vel**2)

    # # Potential Energy:

    # # positions r = [x,y,z] for all particles
    # x = pos[:, 0:1]
    # y = pos[:, 1:2]
    # z = pos[:, 2:3]

    # # matrix that stores all pairwise particle separations: r_j - r_i
    # dx = x.T - x
    # dy = y.T - y
    # dz = z.T - z

    # # matrix that stores 1/r for all particle pairwise particle separations
    # inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    # inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # # sum over upper triangle, to count each interaction only once
    # # PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
    # PE = G * np.sum(np.triu(-(mass * mass.T) * inv_r, 1))

    # return KE, PE

    # ---------------------------------------------------------------#

    pid_i = tl.program_id(0)  # block index over i
    pid_j = tl.program_id(1)  # block index over j

    offs_i = pid_i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_j = pid_j * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_i = offs_i < N
    mask_j = offs_j < N

    # Load positions for i, j
    xi = tl.load(pos + offs_i * 3 + 0, mask=mask_i, other=0.0)
    yi = tl.load(pos + offs_i * 3 + 1, mask=mask_i, other=0.0)
    zi = tl.load(pos + offs_i * 3 + 2, mask=mask_i, other=0.0)

    xj = tl.load(pos + offs_j * 3 + 0, mask=mask_j, other=0.0)
    yj = tl.load(pos + offs_j * 3 + 1, mask=mask_j, other=0.0)
    zj = tl.load(pos + offs_j * 3 + 2, mask=mask_j, other=0.0)

    mi = tl.load(mass + offs_i, mask=mask_i, other=0.0)
    mj = tl.load(mass + offs_j, mask=mask_j, other=0.0)

    # Broadcast indices to figure out which global pairs (i,j) we are
    ii = offs_i[:, None]   # [BLOCK_SIZE_N, 1]
    jj = offs_j[None, :]   # [1, BLOCK_SIZE_K]

    # Pairwise separations r_j - r_i, like x.T - x, etc.
    dx = xj[None, :] - xi[:, None]
    dy = yj[None, :] - yi[:, None]
    dz = zj[None, :] - zi[:, None]

    # |r_j - r_i|
    r2 = dx * dx + dy * dy + dz * dz
    r = tl.sqrt(r2)

    # Only consider valid entries: indices in range, and upper triangle i<j
    mask_pairs = (
        (ii < N) & (jj < N) & (jj > ii) & (r > 0)
    )  # boolean [BLOCK_SIZE_N, BLOCK_SIZE_K]

    inv_r = tl.where(mask_pairs, 1.0 / r, 0.0)

    # Mass products m_i * m_j
    mi_2d = mi[:, None]      # [BLOCK_SIZE_N, 1]
    mj_2d = mj[None, :]      # [1, BLOCK_SIZE_K]
    mm = mi_2d * mj_2d       # [BLOCK_SIZE_N, BLOCK_SIZE_K]

    # energy contribution per pair: -G * m_i m_j / r_ij
    tile_energy = -G * mm * inv_r

    # Sum over this tile
    tile_sum = tl.sum(tile_energy, axis=0)
    tile_sum = tl.sum(tile_sum, axis=0)  # scalar

    # Atomically add into global accumulator
    tl.atomic_add(pe, tile_sum)
   

def nbody(mass, pos, vel, N, Nt, dt, G, softening):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    vel is N x 3 matrix of velocities
    """
    # Make sure N, Nt, G, softening, dt are plain Python scalars
    N = int(N)
    Nt = int(Nt)
    G = float(G)
    softening = float(softening)
    dt = float(dt)

    # Get DTYPE and assert dtypes
    assert mass.dtype == pos.dtype == vel.dtype, "mass, pos, and vel must have the same dtype"
    dtype = pos.dtype
    assert dtype in (torch.float32, torch.float64)
    DTYPE = tl.float32 if dtype == torch.float32 else tl.float64

    # define grids for kernel launches
    grid_1d = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    grid_2d = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                            triton.cdiv(N, meta["BLOCK_SIZE_K"]))

    # Convert to Center-of-Mass frame
    # vel -= np.mean(mass * vel, axis=0) / np.mean(mass)
    mom = (mass * vel).mean(dim=0)   # shape (3,)
    m_mean = mass.mean()             # scalar

    vel -= mom / m_mean

    # calculate initial gravitational accelerations
    # acc = getAcc(pos, mass, G, softening)
    acc = torch.zeros((N, 3), dtype=pos.dtype)
    _get_acc[grid_1d](pos, mass, G, softening, acc, N, DTYPE)

    # # calculate initial energy of system
    # KE = np.ndarray(Nt + 1, dtype=mass.dtype)
    # PE = np.ndarray(Nt + 1, dtype=mass.dtype)
    # KE[0], PE[0] = getEnergy(pos, vel, mass, G)
    KE = torch.empty(Nt + 1, dtype = dtype)
    PE = torch.empty(Nt + 1, dtype = dtype)
    pe_acc = torch.zeros((1,), dtype=dtype)
    _get_energy[grid_2d](pos, mass, G, pe_acc, N, DTYPE)
    KE[0] = 0.5 * torch.sum(mass * vel**2) 
    PE[0] = pe_acc[0]

    # Main loop
    t = 0.0
    for i in range(Nt):
        # 1/2 kick
        vel += acc * (dt / 2.0)

        # drift
        pos += vel * dt

        # update accelerations
        _get_acc[grid_1d](pos, mass, G, softening, acc, N, DTYPE)

        # 1/2 kick
        vel += acc * (dt / 2.0)
        t += dt

        # get energy of system
        pe_acc.zero_()
        _get_energy[grid_2d](pos, mass, G, pe_acc, N, DTYPE)
        KE[i + 1] = 0.5 * torch.sum(mass * vel**2)
        PE[i + 1] = pe_acc[0]

    return KE, PE
