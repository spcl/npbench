import jax
import jax.numpy as jnp
from jax import lax

def kernel(alpha, beta, C: jax.Array, A: jax.Array, B: jax.Array):

    # Scale C by beta
    C = C * beta

    def row_update(i, C_and_temp2):
        C, _ = C_and_temp2
        temp2 = jnp.zeros((C.shape[1],), dtype=C.dtype)

        def col_update(j, val):
            C, temp2 = val

            # Create a mask for elements up to `i`
            mask = jnp.arange(C.shape[0]) < i

            # Masked elements to simulate A[i, :i] and B[:i, j]
            A_slice = jnp.where(mask, A[i, :], 0)  # A[i, :i] with mask
            B_slice = jnp.where(mask, B[:, j], 0)  # B[:i, j] with mask

            # Equivalent to C[:i, j] += alpha * B[i, j] * A[i, :i]
            C = C.at[:, j].add(alpha * B[i, j] * A_slice)

            # Set temp2[j] with the masked dot product result
            temp2 = temp2.at[j].set(jnp.dot(B_slice, A_slice))
            return C, temp2

        # Update columns in row `i`
        C, temp2 = lax.fori_loop(0, C.shape[1], col_update, (C, temp2))

        # Update row `i` after column updates
        C = C.at[i, :].add(alpha * B[i, :] * A[i, i] + alpha * temp2)

        return C, temp2

    # Apply the row update across all rows
    C, _ = lax.fori_loop(0, C.shape[0], row_update, (C, jnp.zeros(C.shape[1], dtype=C.dtype)))
    return C

