import dpnp as np
import numba_dpex as dpex

@dpex.kernel
def build_up_b(b, rho, dt, u, v, dx, dy):
    for i in range(1, b.shape[0] - 1):
        for j in range(1, b.shape[1] - 1):
            b[i, j] = (
                rho * (
                    (1 / dt * ((u[i, j + 1] - u[i, j - 1]) / (2 * dx) +
                               (v[i + 1, j] - v[i - 1, j]) / (2 * dy))) -
                    ((u[i, j + 1] - u[i, j - 1]) / (2 * dx))**2 -
                    2 * ((u[i + 1, j] - u[i - 1, j]) / (2 * dy) *
                         (v[i, j + 1] - v[i, j - 1]) / (2 * dx)) -
                    ((v[i + 1, j] - v[i - 1, j]) / (2 * dy))**2
                )
            )

@dpex.kernel
def pressure_poisson(nit, p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nit):
        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                pn[i, j] = p[i, j]  # Copy for updates

        for i in range(1, p.shape[0] - 1):
            for j in range(1, p.shape[1] - 1):
                p[i, j] = (
                    ((pn[i, j + 1] + pn[i, j - 1]) * dy**2 +
                     (pn[i + 1, j] + pn[i - 1, j]) * dx**2) /
                    (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /
                    (2 * (dx**2 + dy**2)) * b[i, j]
                )
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2

@dpex.kernel
def cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un[:] = u.copy()
        vn[:] = v.copy()

        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(nit, p, dx, dy, b)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy *
            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) *
            (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu *
            (dt / dx**2 *
             (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
             dt / dy**2 *
             (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
            (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
            vn[1:-1, 1:-1] * dt / dy *
            (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) *
            (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
            (dt / dx**2 *
             (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
             dt / dy**2 *
             (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        )

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

