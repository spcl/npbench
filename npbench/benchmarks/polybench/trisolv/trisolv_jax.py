import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(L, x, b):

    def loop_body(i, loop_vars):
        L, x, b = loop_vars
        mask = jnp.arange(x.shape[0]) < i
        products = jnp.where(mask, L[i, :] * x, 0.0) 
        dot_product = jnp.sum(products)
        x.at[i].set((b[i] - dot_product) / L[i, i])
        return L, x, b

    _, x, _ = lax.fori_loop(0, x.shape[0], loop_body, (L, x, b))

    return x
