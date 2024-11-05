import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnums=(0, 1, 2))
def kernel(NR: int, NQ: int, NP: int, A:jax.Array, C4:jax.Array):

    # for r in range(NR):
    #     for q in range(NQ):
    #         sum[:] = A[r, q, :] @ C4
    #         A[r, q, :] = sum
    A = A.at[:].set(jnp.reshape(jnp.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP)))
    return A
