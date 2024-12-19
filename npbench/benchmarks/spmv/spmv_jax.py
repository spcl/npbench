# Sparse Matrix-Vector Multiplication (SpMV)
from jax.experimental import sparse
import scipy

# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
def spmv(A_row, A_col, A_val, x):
    dim = A_row.size - 1 # needed because for the "paper" test size, scipy auto-infers the dims wrong
    csr_m = scipy.sparse.csr_matrix((A_val, A_col, A_row), shape=(dim, dim))

    bcoo_m = sparse.BCOO.from_scipy_sparse(csr_m)
    return bcoo_m @ x
