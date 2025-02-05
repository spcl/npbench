import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0, 1, 2))
def contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y):
    P0 = jnp.zeros((NR, NM), dtype=jnp.complex128)
    P1 = jnp.zeros((NR, NM), dtype=jnp.complex128)

    def body_fun(i, accum):
        P0, P1 = accum
        z = int_pts[i]
        Tz = jnp.zeros((NR, NR), dtype=jnp.complex128)

        def compute_Tz(n, Tz):
            zz = jnp.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
            return Tz

        Tz = jax.lax.fori_loop(0, slab_per_bc + 1, compute_Tz, Tz)

        if NR == NM:
            X = jnp.linalg.inv(Tz)
        else:
            X = jnp.linalg.solve(Tz, Y)
        X = jax.lax.cond(abs(z) < 1.0, lambda x: -x, lambda x: x, X)

        P0 += X
        P1 += z * X

        return P0, P1

    P0, P1 = jax.lax.fori_loop(0, int_pts.shape[0], body_fun, (P0, P1))

    return P0, P1
