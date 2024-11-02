import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(A: jax.Array):
    
    # Set A[0, 0] to its square root
    A = A.at[0, 0].set(jnp.sqrt(A[0, 0]))

    def row_update(i, A):
        
        def col_update(j, A):
            # Create a mask for elements up to `j`
            mask = jnp.arange(A.shape[1]) < j

            # Equivalent of A[i, :j] and A[j, :j] using masking
            A_i_slice = jnp.where(mask, A[i, :], 0)
            A_j_slice = jnp.where(mask, A[j, :], 0)

            # A[i, j] -= dot(A[i, :j], A[j, :j])
            dot_product = jnp.dot(A_i_slice, A_j_slice)
            A = A.at[i, j].set(A[i, j] - dot_product)

            # A[i, j] /= A[j, j]
            A = A.at[i, j].divide(A[j, j])

            return A

        # Column update for all `j` in range(0, i)
        A = lax.fori_loop(0, i, col_update, A)

        # Equivalent of A[i, i] -= dot(A[i, :i], A[i, :i])
        A_i_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0)
        dot_product = jnp.dot(A_i_slice, A_i_slice)
        A = A.at[i, i].set(A[i, i] - dot_product)

        # Set A[i, i] to its square root
        A = A.at[i, i].set(jnp.sqrt(A[i, i]))

        return A

    # Apply row update for all `i` in range(1, A.shape[0])
    A = lax.fori_loop(1, A.shape[0], row_update, A)

    return A
