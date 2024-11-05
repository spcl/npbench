import jax
import jax.numpy as jnp


@jax.jit
def kernel(alpha, beta, C, A):

    for i in range(A.shape[0]):
        C = C.at[i, :i + 1].set(C[i, :i + 1] * beta)
        for k in range(A.shape[1]):
            C = C.at[i, :i + 1].set(C[i, :i + 1] + alpha * A[i, k] * A[:i + 1, k])

    return C
