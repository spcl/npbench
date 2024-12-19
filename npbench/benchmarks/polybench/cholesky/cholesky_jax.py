import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(A):

    A = A.at[0, 0].set(jnp.sqrt(A[0, 0]))

    def row_update(i, A):

        def col_update(j, A):
            mask = jnp.arange(A.shape[1]) < j

            A_i_slice = jnp.where(mask, A[i, :], 0)
            A_j_slice = jnp.where(mask, A[j, :], 0)

            dot_product = jnp.dot(A_i_slice, A_j_slice)
            A = A.at[i, j].set((A[i, j] - dot_product) / A[j, j])

            return A

        A = lax.fori_loop(0, i, col_update, A)

        A_i_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0)
        dot_product = jnp.dot(A_i_slice, A_i_slice)
        A = A.at[i, i].set(jnp.sqrt(A[i, i] - dot_product))

        return A

    A = lax.fori_loop(1, A.shape[0], row_update, A)

    return A
