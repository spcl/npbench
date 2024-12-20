import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):

    def loop_body(i, corr):
        corr.at[i, i].set(1)
        return corr

    mean = jnp.mean(data, axis=0)
    stddev = jnp.std(data, axis=0)
    stddev = jnp.where(stddev <= 0.1, 1.0, stddev)
    data = data - mean
    data = data / (jnp.sqrt(float_n) * stddev)

    corr = jnp.dot(data.T, data)

    corr = lax.fori_loop(0, M, loop_body, corr)

    return corr
