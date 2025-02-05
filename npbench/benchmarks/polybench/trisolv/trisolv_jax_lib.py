import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(L, x, b):
    
    x = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    return x
