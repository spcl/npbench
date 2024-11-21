import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):


    def loop_body(i, loop_vars):
        corr, data = loop_vars
        corr = lax.dynamic_update_slice(corr, jnp.roll(data[i, None], -(i + 1), axis=1), (i, i + 1))
        corr = lax.dynamic_update_slice(corr, jnp.roll(data[i, None], -(i + 1), axis=1).T, (i + 1, i))

        return corr, data

    mean = jnp.mean(data, axis=0)
    stddev = jnp.std(data, axis=0)
    stddev = jnp.where(stddev <= 0.1, 1.0, stddev)
    data = data - mean
    data = data / (jnp.sqrt(float_n) * stddev)
    corr = jnp.eye(M, dtype=data.dtype)

    data_mul = jnp.dot(data.T, data)

    corr, _ = lax.fori_loop(0, M - 1, loop_body, (corr, data_mul))

    return corr
