import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):
    
    mean = jnp.mean(data, axis=0)
    data -= mean
    cov = data.T @ data / (float_n - 1.0)
    return cov
