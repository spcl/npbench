import itertools
import torch
import triton
import triton.language as tl

def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if m != 128 or n != 128
    ]

def generate_config_col():
    return [
        triton.Config(kwargs={"BLOCK_SIZE": m}, num_warps=w)
        for m, w in itertools.product(
            [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
    ]

@triton.autotune(configs=generate_config_col(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_div_column(
    A_ptr, stride_am, stride_an,
    N, k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Divide the column below the pivot:
        for i in k+1..N-1: A[i,k] /= A[k,k]
    """
    pid = tl.program_id(axis=0)
    rows = k + 1 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = k

    # pivot
    pivot_ptr = A_ptr + k * stride_am + k * stride_an
    pivot = tl.load(pivot_ptr)

    # column slice to scale
    col_ptrs = A_ptr + rows * stride_am + col * stride_an
    mask = rows < N
    vals = tl.load(col_ptrs, mask=mask, other=0.0)
    vals = vals / pivot
    tl.store(col_ptrs, vals, mask=mask)

@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_trailing_update(
    A_ptr, stride_am, stride_an,
    N, k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Rank-1 update on trailing submatrix:
        A[k+1:, k+1:] -= A[k+1:, k] @ A[k, k+1:]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rows = k + 1 + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = k + 1 + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = rows[:, None] < N
    cn = cols[None, :] < N
    mask = rm & cn

    # pointers
    a_ptrs = A_ptr + rows[:, None] * stride_am + cols[None, :] * stride_an
    l_ptrs = A_ptr + rows * stride_am + k * stride_an          # L column (k)
    u_ptrs = A_ptr + k * stride_am + cols * stride_an          # U row (k)

    Ablk = tl.load(a_ptrs, mask=mask, other=0.0)
    Lcol = tl.load(l_ptrs, mask=rows < N, other=0.0)[:, None]
    Urow = tl.load(u_ptrs, mask=cols < N, other=0.0)[None, :]

    # rank-1 update
    Aupd = Ablk - Lcol * Urow

    tl.store(a_ptrs, Aupd, mask=mask)


def kernel(A: torch.Tensor):
    """
    LU factorization
    On return, A has L (unit diag, below diag) and U (on/above diag).
    """
    N = A.shape[0]

    stride_am, stride_an = A.stride()

    for k in range(N):
        # 1) scale column below pivot
        grid_col = lambda meta: (
            triton.cdiv((max(N - (k + 1), 0) + meta["BLOCK_SIZE"] - 1), meta["BLOCK_SIZE"]),
        )
        _kernel_lu_div_column[grid_col](
            A, stride_am, stride_an,
            N, k,
        )

        # 2) update trailing submatrix
        rem = N - (k + 1)
        if rem <= 0:
            continue
        grid = lambda meta: (
            triton.cdiv((rem + meta["BLOCK_SIZE_M"] - 1), meta["BLOCK_SIZE_M"]),
            triton.cdiv((rem + meta["BLOCK_SIZE_N"] - 1), meta["BLOCK_SIZE_N"]),
        )
        _kernel_lu_trailing_update[grid](
            A, stride_am, stride_an,
            N, k,
        )

    return A

