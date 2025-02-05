import jax.numpy as jnp
import jax
from jax import lax
from functools import partial

@jax.jit
def match(b1, b2):
    return jnp.where(b1 + b2 == 3, 1, 0)


@partial(jax.jit, static_argnums=(0,))
def kernel(N, seq):

    table = jnp.zeros((N, N), jnp.int32)

    def func_i(i, table):
        i = N - 1 - i
        def func_j(j, table):
            table = table.at[i, j].set(
                jnp.where(
                    j - 1 >= 0, 
                    jnp.maximum(table[i, j], table[i, j - 1]), 
                    table[i, j]
                )
            )
            table = table.at[i, j].set(
                jnp.where(
                    i + 1 < N, 
                    jnp.maximum(table[i, j], table[i + 1, j]), 
                    table[i, j]
                )
            )
            table = table.at[i, j].set(
                jnp.where(
                    (j - 1 >= 0) & (i + 1 < N) & (i < j - 1),
                    jnp.maximum(table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j])),
                    table[i, j]
                )
            )
            table = table.at[i, j].set(
                jnp.where(
                    (j - 1 >= 0) & (i + 1 < N) & (i >= j - 1),
                    jnp.maximum(table[i, j], table[i + 1, j - 1]),
                    table[i, j]
                )
            )

            def func_k(k, table):
                table = table.at[i, j].set(
                    jnp.maximum(
                        table[i, j],
                        table[i, k] + table[k + 1, j]
                    )
                )
                return table

            table = lax.fori_loop(i + 1, j, func_k, table)
            return table

        table = lax.fori_loop(i + 1, N, func_j, table)
        return table

    table = lax.fori_loop(0, N, func_i, table)
    return table
