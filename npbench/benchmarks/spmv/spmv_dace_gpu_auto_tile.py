# Sparse Matrix-Vector Multiplication (SpMV)
import numpy as np
import dace as dc

M, N, nnz = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N', 'nnz'))


# Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
# (CSR) format
@dc.program
def _spmv(A_row: dc.uint32[M + 1], A_col: dc.uint32[nnz],
         A_val: dc.float64[nnz], x: dc.float64[N]):
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

_best_config = None

def autotuner(A_row, A_col, A_val, x, M, N, nnz):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _spmv.to_sdfg(),
        {"A_row": A_row, "A_col": A_col, "A_val": A_val, "x": x, "M": M, "N":N, "nnz":nnz},
        dims=get_max_ndim([A_row, A_col, A_val, x])
    )

def spmv(A_row, A_col, A_val, x, M, N, nnz):
    global _best_config
    y = _best_config(A_row, A_col, A_val, x, M, N, nnz)
    return y
