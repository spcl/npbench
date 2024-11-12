import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):

    mean = jnp.mean(data, axis=0)
    stddev = jnp.std(data, axis=0)
    stddev = jnp.where(stddev <= 0.1, 1.0, stddev)
    data = data - mean
    data = data / (jnp.sqrt(float_n) * stddev)
    corr = jnp.eye(M, dtype=data.dtype)

    for i in range(M - 1):
        corr = corr.at[i + 1:M, i].set(data[:, i] @ data[:, i + 1:M]) 
        corr = corr.at[i, i + 1:M].set(data[:, i] @ data[:, i + 1:M])

    return corr
