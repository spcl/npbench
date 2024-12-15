import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def kernel(M, float_n, data):
    
    mean = jnp.mean(data, axis=0)
    data -= mean

    cov = jnp.zeros((M, M), dtype=data.dtype)

    def loop_body(i, loop_vars):
        data, cov = loop_vars
        cov_slice1 = data[:, i]
        cov_slice2 = jnp.where(jnp.arange(M) >= i, data, 0.0)

        ans = cov_slice1 @ cov_slice2 / (float_n - 1.0)
        row_update_slice = jnp.where(jnp.arange(M) >= i, ans, cov[i, :])
        col_update_slice = jnp.where(jnp.arange(M) >= i, ans, cov[:, i])

        cov = cov.at[i, :].set(row_update_slice)
        cov = cov.at[:, i].set(col_update_slice)

        return data, cov
    
    _, cov = lax.fori_loop(0, M, loop_body, (data, cov))
    
    return cov
