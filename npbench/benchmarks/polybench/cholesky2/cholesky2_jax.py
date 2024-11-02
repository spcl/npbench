import jax
import jax.numpy as jnp

@jax.jit
def kernel(A: jax.Array):
    A = A.at[:].set(jnp.linalg.cholesky(A) + jnp.triu(A, k=1))

    return A
