import jax
import jax.numpy as jnp

@jax.jit
def kernel(alpha, beta, A: jax.Array, B: jax.Array, x: jax.Array):

    return (alpha * A + beta * B) @ x
