import jax
import jax.numpy as jnp


@jax.jit
def kernel(alpha: jnp.float64, beta: jnp.float64, A: jax.Array, B: jax.Array, C: jax.Array, D: jax.Array):

    D = D.at[:].set(alpha * A @ B @ C + beta * D)
    return D
