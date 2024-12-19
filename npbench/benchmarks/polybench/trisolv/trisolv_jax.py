import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(L, x, b):
    
    def loop_body(i, x):
        mask = jnp.arange(L.shape[1]) < i
        L_slice = jnp.where(mask, L[i, :], 0.0)
        x_slice = jnp.where(mask, x, 0.0)

        return x.at[i].set((b[i] - jnp.dot(L_slice, x_slice)) / L[i, i])

    return lax.fori_loop(0, x.shape[0], loop_body, x)
