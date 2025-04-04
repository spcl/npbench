import jax
import jax.numpy as jnp


@jax.jit
def kernel(alpha, beta, A, B, C, D):

    D = D.at[:].set(alpha * A @ B @ C + beta * D)
    return D
