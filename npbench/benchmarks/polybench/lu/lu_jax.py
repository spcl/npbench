import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(A):

    def loop_body(i, A):

        def inner_loop_1(j, A):
            A_slice_1 = jnp.where(jnp.arange(A.shape[1]) < j, A[i, :], 0.0)
            A_slice_2 = jnp.where(jnp.arange(A.shape[0]) < j, A[:, j], 0.0)
            A = A.at[i, j].set(A[i, j] - A_slice_1 @ A_slice_2)
            A = A.at[i, j].set(A[i, j] / A[j, j])
            return A
        
        def inner_loop_2(j, A):
            A_slice_1 = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0.0)
            A_slice_2 = jnp.where(jnp.arange(A.shape[0]) < i, A[:, j], 0.0)
            A = A.at[i, j].set(A[i, j] - A_slice_1 @ A_slice_2)
            return A
        
        A = lax.fori_loop(0, i, inner_loop_1, A)
        A = lax.fori_loop(i, A.shape[0], inner_loop_2, A)

        return A
    
    A = lax.fori_loop(0, A.shape[0], loop_body, A)
    
    return A
