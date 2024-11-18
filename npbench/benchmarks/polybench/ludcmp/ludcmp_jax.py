import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def kernel(A, b):
    
    x = jnp.zeros_like(b)
    y = jnp.zeros_like(b)

    def loop_body_1(i, A):
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

    def loop_body_2(i, loop_vars):
        A, y, b = loop_vars
        A_slice = jnp.where(jnp.arange(A.shape[1]) < i, A[i, :], 0.0)
        y_slice = jnp.where(jnp.arange(y.shape[0]) < i, y, 0.0)
        y = y.at[i].set(b[i] - A_slice @ y_slice)
        return A, y, b
    
    def loop_body_3(t, loop_vars):
        A, x, y = loop_vars
        i = A.shape[0] - 1 - t
        A_slice = jnp.where(jnp.arange(A.shape[1]) > i, A[i, :], 0.0)
        x_slice = jnp.where(jnp.arange(x.shape[0]) > i, x, 0.0)
        x = x.at[i].set((y[i] - A_slice @ x_slice) / A[i, i])
        return A, x, y
    
    A = lax.fori_loop(0, A.shape[0], loop_body_1, A)
    A, y, b = lax.fori_loop(0, A.shape[0], loop_body_2, (A, y, b))
    A, x, y = lax.fori_loop(0, A.shape[0], loop_body_3, (A, x, y))

    return x, y
