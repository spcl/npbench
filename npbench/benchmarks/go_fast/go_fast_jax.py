# https://numba.readthedocs.io/en/stable/user/5minguide.html

import jax
import jax.numpy as jnp

@jax.jit
def go_fast(a: jax.Array):
    trace = 0.0
    def body_fn(i, trace):
        trace += jnp.tanh(a[i, i])
        return trace
    trace = jax.lax.fori_loop(0, a.shape[0], body_fn, trace)
    return a + trace
