import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def kernel(TSTEPS, N, A):

    def loop1(t, A):

        def loop2(i, A):
            
            def loop3(j, A):
                A = A.at[i, j].set((A[i, j] + A[i, j - 1]) / 9.0)
                return A
            
            A = A.at[i, 1:-1].set(
                A[i, 1:-1] + (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                            A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                            A[i + 1, 2:])
            )

            A = lax.fori_loop(1, N - 1, loop3, A)
            return A
        
        A = lax.fori_loop(1, N - 1, loop2, A)
        return A

    A = lax.fori_loop(0, TSTEPS - 1, loop1, A)
    
    return A
