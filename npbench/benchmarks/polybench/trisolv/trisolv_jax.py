import jax
import jax.numpy as jnp


@jax.jit
def kernel(L, x, b):

    def loop_body(i, loop_vars):
        L, x, b = loop_vars
        L_slice = jnp.where(jnp.arange(L.shape[1]) < i, L[i, :], 0.0)
        x_slice = jnp.where(jnp.arange(x.shape[0]) < i, x, 0.0)
        x = x.at[i].set((b[i] - L_slice @ x_slice) / L[i, i])
        return L, x, b

    _, x, _ = jax.lax.fori_loop(0, x.shape[0], loop_body, (L, x, b))

    return x
