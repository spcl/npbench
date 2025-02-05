# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=["xn", "yn", "maxiter"])
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = jnp.linspace(xmin, xmax, xn, dtype=jnp.float64)
    Y = jnp.linspace(ymin, ymax, yn, dtype=jnp.float64)
    C = X + Y[:, None] * 1j
    N = jnp.zeros(C.shape, dtype=jnp.int64)
    Z = jnp.zeros(C.shape, dtype=jnp.complex128)

    def body_fun(n, state):
        Z, N = state
        I = jnp.less(jnp.abs(Z), horizon)
        new_N = jnp.where(I, n, N)
        new_Z = jnp.where(I, Z**2 + C, Z)
        return new_Z, new_N

    Z, N = jax.lax.fori_loop(0, maxiter, body_fun, (Z, N))
    N = jnp.where(N == maxiter-1, 0, N)
    return Z, N
