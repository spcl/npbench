import jax
import jax.numpy as jnp


@jax.jit
def kernel(M, float_n, data):
    
    cov = jnp.cov(data, rowvar=False)

    return cov
