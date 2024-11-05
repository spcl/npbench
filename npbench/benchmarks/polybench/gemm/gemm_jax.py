import jax
import jax.numpy as jnp

@jax.jit
def kernel(alpha: jnp.float64, beta: jnp.float64, C:jax.Array, A:jax.Array, B:jax.Array):

    C = C.at[:].set(alpha * A @ B + beta * C)
    return C
