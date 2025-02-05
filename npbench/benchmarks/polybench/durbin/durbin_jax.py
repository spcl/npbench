import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(r):

    y = jnp.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y = y.at[0].set(-r[0])

    def loop_body(k, loop_vars):
        alpha, beta, y, r = loop_vars
        beta *= 1.0 - alpha * alpha
        mask = jnp.arange(r.shape[0]) < k

        products = jnp.where(mask, y * jnp.roll(jnp.flip(r), [k], 0),0.0)
        dot_prod = jnp.sum(products)
        alpha = -(r[k] + dot_prod) / beta

        y_update_slice = jnp.where(mask, jnp.roll(jnp.flip(y), [k], 0) * alpha, 0.0)
        y += y_update_slice
        y = y.at[k].set(alpha)

        return alpha, beta, y, r

    _, _, y, _ = lax.fori_loop(1, r.shape[0], loop_body, (alpha, beta, y, r))

    return y
