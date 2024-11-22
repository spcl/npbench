import jax
import jax.numpy as jnp

@jax.jit
def kernel(alpha, beta, C, A, B):

    C = C.at[:].set(alpha * A @ B + beta * C)
    return C
