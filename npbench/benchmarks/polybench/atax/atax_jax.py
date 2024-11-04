import jax
import jax.numpy as jnp

@jax.jit
def kernel(A: jax.Array, x: jax.Array):
    return (A @ x) @ A
