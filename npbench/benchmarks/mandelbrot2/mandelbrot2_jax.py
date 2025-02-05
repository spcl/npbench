# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=["xn", "yn", "itermax"])
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = jnp.mgrid[0:xn, 0:yn]
    X = jnp.linspace(xmin, xmax, xn, dtype=jnp.float64)[Xi]
    Y = jnp.linspace(ymin, ymax, yn, dtype=jnp.float64)[Yi]
    C = X + Y * 1j
    N_ = jnp.zeros(C.shape, dtype=jnp.int64)
    Z_ = jnp.zeros(C.shape, dtype=jnp.complex128)

    original_shape = C.shape
    Xi = Xi.reshape(-1)
    Yi = Yi.reshape(-1)
    C = C.reshape(-1)

    def body_fun(i, state):
        Z, Xi, Yi, C, N_, Z_, mask = state
        # Compute for relevant points only
        Z = Z * Z + C

        # Failed convergence
        I = abs(Z) > horizon
        I = I & mask  # Only consider points that haven't diverged yet

        N_ = jnp.where(I, i + 1, N_)
        Z_ = jnp.where(I, Z, Z_)

        # Keep going with those who have not diverged yet
        mask = mask & ~I
        Z = jnp.where(mask, Z, 0)

        return (Z, Xi, Yi, C, N_, Z_, mask)

    init_state = (jnp.zeros_like(C, dtype=jnp.complex128), Xi, Yi, C,
                  N_.reshape(-1), Z_.reshape(-1), jnp.ones_like(C, dtype=bool))
    _, _, _, _, N_, Z_, _ = jax.lax.fori_loop(0, itermax, body_fun, init_state)

    Z_ = Z_.reshape(original_shape) # Reshape results back to original shape
    N_ = N_.reshape(original_shape)

    return Z_.T, N_.T
