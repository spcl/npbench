import jax
import jax.numpy as jnp


@jax.jit
def kernel(path):

    for k in range(path.shape[0]):
        path = path.at[:].set(jnp.minimum(path[:], jnp.add.outer(path[:, k], path[k, :])))

    return path