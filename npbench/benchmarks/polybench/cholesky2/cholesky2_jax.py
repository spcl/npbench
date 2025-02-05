import jax
import jax.numpy as jnp

@jax.jit
def kernel(A: jax.Array):

    L = jnp.linalg.cholesky(A)
    upper_A = jnp.triu(A, k=1)

    return L + upper_A
