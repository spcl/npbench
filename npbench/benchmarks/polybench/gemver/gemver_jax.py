import jax
import jax.numpy as jnp


@jax.jit
def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    
    A += jnp.outer(u1, v1) + jnp.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x

    return A, x, w