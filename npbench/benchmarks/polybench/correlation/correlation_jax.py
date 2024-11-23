import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):


    def loop_body(i, loop_vars):
        corr, data = loop_vars
        corr_update_x = jnp.where(jnp.arange(data.shape[0]) > i, data[i], corr[i])
        corr_update_y = jnp.where(jnp.arange(data.shape[0]) > i, data[i], corr[:, i])
        corr = lax.dynamic_update_slice(corr, corr_update_x[None, :], (i, 0))
        corr = lax.dynamic_update_slice(corr, corr_update_y[:, None], (0, i))

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
