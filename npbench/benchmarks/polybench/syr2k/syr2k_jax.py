import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(alpha, beta, C, A, B):

    def loop_body(i, loop_vars):
        
        def inner_loop(k, loop_vars):
            alpha, C, A, B = loop_vars
            A_update_slice = jnp.where(jnp.arange(A.shape[0]) < i + 1, A[:, k], 0.0)
            A_update_slice *= alpha * B[i, k]


            B_update_slice = jnp.where(jnp.arange(B.shape[0]) < i + 1, B[:, k], 0.0)
            B_update_slice *= alpha * A[i, k]

            C_update_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C[i, :], 0.0)
            C_update_slice += A_update_slice + B_update_slice
            C_update_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C_update_slice, C[i, :])

            C = lax.dynamic_update_slice(C, C_update_slice[None, :], (i, 0))
            return alpha, C, A, B
        
        
        alpha, beta, C, A, B = loop_vars
        C_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C[i, :], 0.0)
        C_slice = C_slice * beta
        C_slice = jnp.where(jnp.arange(C.shape[1]) < i + 1, C_slice, C[i, :])
        
        C = lax.dynamic_update_slice(C, C_slice[None, :], (i, 0))

        _, C, _, _ = lax.fori_loop(0, A.shape[1], inner_loop, (alpha, C, A, B))

        return alpha, beta, C, A, B

    _, _, C, _, _ = lax.fori_loop(0, A.shape[0], loop_body, (alpha, beta, C, A, B))
            
    return C
