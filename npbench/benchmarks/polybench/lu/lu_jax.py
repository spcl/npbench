import jax
import jax.numpy as jnp


@jax.jit
def kernel(A):
    
    for i in range(A.shape[0]):
        for j in range(i):
            A = A.at[i, j].set(A[i, j] - A[i, :j] @ A[:j, j])
            A = A.at[i, j].set(A[i, j] / A[j, j])
        for j in range(i, A.shape[0]):
            A = A.at[i, j].set(A[i, j] - A[i, :i] @ A[:i, j])

    return A
