# Sparse Matrix-Vector Multiplication (SpMV)
import jax.numpy as jnp
import jax
from jax import lax

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@jax.jit
def spmv(A_row, A_col, A_val, x):
    y = jnp.empty(A_row.size - 1, dtype=A_val.dtype)

    def row_update(i, y):
        mask = (jnp.arange(A_col.size) >= A_row[i]) & (jnp.arange(A_col.size) < A_row[i + 1])

        cols = jnp.where(mask, A_col, 0)
        vals = jnp.where(mask, A_val, 0)
        y = y.at[i].set(vals @ x[cols])

        return y

    y = lax.fori_loop(0, A_row.size - 1, row_update, y)
    return y
