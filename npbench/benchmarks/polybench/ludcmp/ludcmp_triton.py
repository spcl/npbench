import itertools
import torch
import triton
import triton.language as tl


def generate_config_2d():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [16, 32, 64, 128], [16, 32, 64, 128], [1, 2, 4, 8]
        )
    ]


def generate_config_1d():
    return [
        triton.Config(kwargs={"BLOCK_SIZE": bsz}, num_warps=w)
        for bsz, w in itertools.product([64, 128, 256, 512, 1024], [1, 2, 4, 8])
    ]


@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_div_column(
    A_ptr, stride_am, stride_an,
    N, k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    for i in k+1..N-1: A[i,k] /= A[k,k]
    """
    pid = tl.program_id(axis=0)
    rows = k + 1 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = k

    # pivot
    pivot_ptr = A_ptr + k * stride_am + k * stride_an
    pivot = tl.load(pivot_ptr)

    # column to scale
    col_ptrs = A_ptr + rows * stride_am + col * stride_an
    mask = rows < N
    vals = tl.load(col_ptrs, mask=mask, other=0.0)
    vals = vals / pivot
    tl.store(col_ptrs, vals, mask=mask)


@triton.autotune(configs=generate_config_2d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_trailing_update(
    A_ptr, stride_am, stride_an,
    N, k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    A[k+1:, k+1:] -= A[k+1:, k] @ A[k, k+1:]   (rank-1 update)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rows = k + 1 + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = k + 1 + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rm = rows[:, None] < N
    cn = cols[None, :] < N
    mask = rm & cn

    a_ptrs = A_ptr + rows[:, None] * stride_am + cols[None, :] * stride_an
    l_ptrs = A_ptr + rows * stride_am + k * stride_an          # L col k
    u_ptrs = A_ptr + k * stride_am + cols * stride_an          # U row k

    Ablk = tl.load(a_ptrs, mask=mask, other=0.0)
    Lcol = tl.load(l_ptrs, mask=rows < N, other=0.0)[:, None]
    Urow = tl.load(u_ptrs, mask=cols < N, other=0.0)[None, :]

    Aupd = Ablk - Lcol * Urow
    tl.store(a_ptrs, Aupd, mask=mask)


@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_forward_row(
    A_ptr, stride_am, stride_an,
    b_ptr, y_ptr,
    N, i,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute: y[i] = b[i] - dot(A[i, :i], y[:i])
    L has unit diagonal, so no division here.
    """
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((), dtype=A_ptr.dtype.element_ty)

    # process in tiles of BLOCK_SIZE
    # num full/partial tiles = ceil(i / BLOCK_SIZE)
    num_tiles = (i + BLOCK_SIZE - 1) // BLOCK_SIZE
    for t in range(0, num_tiles):
        cols = t * BLOCK_SIZE + offs
        mask = cols < i
        a_vals = tl.load(A_ptr + i * stride_am + cols * stride_an, mask=mask, other=0.0)
        y_vals = tl.load(y_ptr + cols, mask=mask, other=0.0)
        acc += tl.sum(a_vals * y_vals, axis=0)

    bi = tl.load(b_ptr + i)
    yi = bi - acc
    tl.store(y_ptr + i, yi)


@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_backward_row(
    A_ptr, stride_am, stride_an,
    y_ptr, x_ptr,
    N, i,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute: x[i] = (y[i] - dot(A[i, i+1:], x[i+1:])) / A[i,i]
    """
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((), dtype=A_ptr.dtype.element_ty)

    # length of the suffix
    len_suf = N - (i + 1)
    num_tiles = (len_suf + BLOCK_SIZE - 1) // BLOCK_SIZE

    for t in range(0, num_tiles):
        cols = (i + 1) + t * BLOCK_SIZE + offs
        mask = cols < N
        a_vals = tl.load(A_ptr + i * stride_am + cols * stride_an, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + cols, mask=mask, other=0.0)
        acc += tl.sum(a_vals * x_vals, axis=0)

    yi = tl.load(y_ptr + i)
    aii = tl.load(A_ptr + i * stride_am + i * stride_an)
    xi = (yi - acc) / aii
    tl.store(x_ptr + i, xi)


def kernel(A: torch.Tensor, b: torch.Tensor):
    N = A.shape[0]
    stride_am, stride_an = A.stride()

    # -------- LU factorization (in-place) --------
    for k in range(N):
        # 1) scale column below pivot
        if k + 1 < N:
            grid_col = lambda meta: (
                triton.cdiv((N - (k + 1)), meta["BLOCK_SIZE"]),
            )

            _kernel_lu_div_column[grid_col](
                A, stride_am, stride_an,
                N, k,
            )

        # 2) rank-1 update of trailing block
        rem = N - (k + 1)
        if rem > 0:
            grid_upd = lambda meta: (
                triton.cdiv(rem, meta["BLOCK_SIZE_M"]),
                triton.cdiv(rem, meta["BLOCK_SIZE_N"]),
            )

            _kernel_lu_trailing_update[grid_upd](
                A, stride_am, stride_an,
                N, k,
            )

    # -------- Forward solve Ly=b (unit lower) --------
    y = torch.empty_like(b)
    # we could zero y first but kernels write y[i] directly
    for i in range(N):
        _kernel_forward_row[(1,)](
            A, stride_am, stride_an,
            b, y,
            N, i,
        )

    # -------- Backward solve Ux=y --------
    x = torch.empty_like(b)
    # initialize x with zeros so reading x[i+1:] is safe (kernels fully write xi)
    x.zero_()
    for i in range(N - 1, -1, -1):
        _kernel_backward_row[(1,)](
            A, stride_am, stride_an,
            y, x,
            N, i,
        )

    return x, y
