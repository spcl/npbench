import jax
import jax.numpy as jnp

# import numpy as np



@jax.jit
def kernel(alpha, A, B):

    L = jnp.triu(A, 1)  # 1 excludes the main diagonal
    B += L @ B
    B *= alpha

    return B
