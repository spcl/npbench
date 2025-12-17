# Sparse Matrix-Vector Multiplication (SpMV)
import numpy as np
import dace as dc
from npbench.infrastructure.dace_framework import dc_float

M, N, nnz = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'nnz'))


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@dc.program
def spmv(A_row: dc.uint32[M + 1], A_col: dc.uint32[nnz],
         A_val: dc_float[nnz], x: dc_float[N]):
    # y = np.empty(A_row.size - 1, A_val.dtype)
    y = np.empty(M, A_val.dtype)

    # for i in range(A_row.size - 1):
    for i in range(M):
        start = dc.define_local_scalar(dc.uint32)
        stop = dc.define_local_scalar(dc.uint32)
        start = A_row[i]
        stop = A_row[i + 1]
        # cols = A_col[A_row[i]:A_row[i + 1]]
        # vals = A_val[A_row[i]:A_row[i + 1]]
        cols = A_col[start:stop]
        vals = A_val[start:stop]
        y[i] = vals @ x[cols]

    return y
