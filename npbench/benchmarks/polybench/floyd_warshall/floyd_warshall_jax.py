import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(path):

    def loop_func(k, path):
        path = path.at[:].set(jnp.minimum(path[:], jnp.add.outer(path[:, k], path[k, :])))
        return path
    
    path = lax.fori_loop(0, path.shape[0], loop_func, path)

    return path
