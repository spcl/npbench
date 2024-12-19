import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(A):

    A = A.at[0, 0].set(jnp.sqrt(A[0, 0]))

    def row_update(i, A):

        def col_update(j, A):
            mask = jnp.arange(A.shape[1]) < j
            products = jnp.where(mask, A[i, :] * A[j, :], 0.0)
            dot_prod = jnp.sum(products)
            A = A.at[i, j].set((A[i, j] - dot_prod) / A[j, j])

            return A

        A = lax.fori_loop(0, i, col_update, A)

        mask = jnp.arange(A.shape[1]) < i
        products = jnp.where(mask, A[i, :] * A[i, :], 0)
        dot_product = jnp.sum(products)
        A = A.at[i, i].set(jnp.sqrt(A[i, i] - dot_product))

        return A

    A = lax.fori_loop(1, A.shape[0], row_update, A)

    return A
