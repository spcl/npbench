import jax
import jax.numpy as jnp

@jax.jit
def kernel(A):

    Q = jnp.zeros_like(A)
    R = jnp.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    def body_fun(k, arrays):
        Q, R, A = arrays

        nrm = jnp.dot(A[:, k], A[:, k])
        R = R.at[k, k].set(jnp.sqrt(nrm))
        Q = Q.at[:, k].set(A[:, k] / R[k, k])

        def inner_body_fun(j, arrays):
            Q, R, A = arrays
            R = R.at[k, j].set(jnp.dot(Q[:, k], A[:, j]))
            A = A.at[:, j].add(-Q[:, k] * R[k, j])
            return Q, R, A

        Q, R, A = jax.lax.fori_loop(k + 1, A.shape[1], inner_body_fun, (Q, R, A))
        return Q, R, A

    Q, R, A = jax.lax.fori_loop(0, A.shape[1], body_fun, (Q, R, A))

    return Q, R
