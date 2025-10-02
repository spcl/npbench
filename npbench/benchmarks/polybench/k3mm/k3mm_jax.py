import jax
import jax.numpy as jnp

@jax.jit
def kernel(A, B, C, D):

    return A @ B @ C @ D
