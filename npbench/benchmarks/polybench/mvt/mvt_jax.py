import jax
import jax.numpy as jnp

@jax.jit
def kernel(x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A

    return (x1, x2)
