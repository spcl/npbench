import jax
import jax.numpy as jnp

@jax.jit
def kernel(A: jax.Array, p: jax.Array, r: jax.Array):

    return r @ A, A @ p
