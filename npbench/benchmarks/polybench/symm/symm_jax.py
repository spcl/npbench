import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(alpha, beta, C: jax.Array, A: jax.Array, B: jax.Array):

    temp2 = jnp.empty((C.shape[1], ), dtype=C.dtype)
    C *= beta

    def row_update(i, arrays):
        C, temp2 = arrays

        def col_update(j, val):
            C, temp2 = val

            A_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0.0)
            B_slice = jnp.where(jnp.arange(B.shape[0]) < i, B[:, j], 0.0)
            
            C = lax.dynamic_update_slice(
                C, 
                (C[:,j] + (alpha * B[i, j] * A_slice))[:, None],
                (0, j)
            )
            temp2 = temp2.at[j].set(B_slice @ A_slice)
            return C, temp2

        C, temp2 = lax.fori_loop(0, C.shape[1], col_update, (C, temp2))
        C = C.at[i, :].add(alpha * B[i, :] * A[i, i] + alpha * temp2)
        return C, temp2

    C, _ = lax.fori_loop(0, C.shape[0], row_update, (C, temp2))
    return C

