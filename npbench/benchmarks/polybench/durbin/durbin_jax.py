import jax
import jax.numpy as jnp


@jax.jit
def kernel(r):

    y = jnp.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y = y.at[0].set(-r[0])

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + jnp.dot(jnp.flip(r[:k]), y[:k])) / beta
        y = y.at[:k].add(alpha * jnp.flip(y[:k]))
        y = y.at[k].set(alpha)

    return y
