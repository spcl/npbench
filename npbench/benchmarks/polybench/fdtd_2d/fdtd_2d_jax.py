import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(TMAX, ex, ey, hz, _fict_):

    def loop_body(t, loop_vars):
        ex, ey, hz = loop_vars
        ey = ey.at[0, :].set(_fict_[t])
        ey = ey.at[1:, :].set(ey[1:, :] - 0.5 * (hz[1:, :] - hz[:-1, :]))
        ex = ex.at[:, 1:].set(ex[:, 1:] - 0.5 * (hz[:, 1:] - hz[:, :-1]))
        hz = hz.at[:-1, :-1].set(hz[:-1, :-1] - 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] +
                                 ey[1:, :-1] - ey[:-1, :-1]))
        return ex, ey, hz

    ex, ey, hz = lax.fori_loop(0, TMAX, loop_body, (ex, ey, hz))
    return ex, ey, hz, _fict_
