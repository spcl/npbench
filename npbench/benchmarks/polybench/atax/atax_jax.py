import jax
import jax.numpy as jnp

@jax.jit
def kernel(A, x):
    return (A @ x) @ A
