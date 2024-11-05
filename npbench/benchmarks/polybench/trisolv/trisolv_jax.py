import jax
import jax.numpy as jnp


@jax.jit
def kernel(L, x, b):

    for i in range(x.shape[0]):
        x = x.at[i].set((b[i] - L[i, :i] @ x[:i]) / L[i, i])

    return x
