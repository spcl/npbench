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

# 1) Diagonal update at step k:
#    L[k,k] = sqrt( A[k,k] - sum_{s<k} L[k,s]^2 )
@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def chol_diag_kernel(A_ptr, stride_am, stride_an, N, k, BLOCK_SIZE: tl.constexpr):
    # reduction across s in chunks of BLOCK_S
    acc = tl.zeros((), dtype=A_ptr.dtype.element_ty)
    s0 = 0
    while s0 < k:
        ss = s0 + tl.arange(0, BLOCK_SIZE)
        ms = ss < k
        row_off = k * stride_am
        lk = tl.load(A_ptr + row_off + ss * stride_an, mask=ms, other=0.0)
        acc += tl.sum(lk * lk, axis=0)
        s0 += BLOCK_SIZE
    akk = tl.load(A_ptr + k * stride_am + k * stride_an)
    val = tl.sqrt(akk - acc)
    tl.store(A_ptr + k * stride_am + k * stride_an, val)

# 2) Column update below diagonal at step k:
#    For i>k: L[i,k] = ( A[i,k] - sum_{s<k} L[i,s]*L[k,s] ) / L[k,k]
@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def chol_col_kernel(A_ptr, stride_am, stride_an, N, k,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = k + 1 + pid
    if i >= N:
        return

    # dot( L[i,:k], L[k,:k] )
    acc = tl.zeros((), dtype=A_ptr.dtype.element_ty)
    s0 = 0
    while s0 < k:
        ss = s0 + tl.arange(0, BLOCK_SIZE)
        ms = ss < k
        li = tl.load(A_ptr + i * stride_am + ss * stride_an, mask=ms, other=0.0)
        lk = tl.load(A_ptr + k * stride_am + ss * stride_an, mask=ms, other=0.0)
        acc += tl.sum(li * lk, axis=0)
        s0 += BLOCK_SIZE

    aik = tl.load(A_ptr + i * stride_am + k * stride_an)
    lkk = tl.load(A_ptr + k * stride_am + k * stride_an)
    lik = (aik - acc) / lkk
    tl.store(A_ptr + i * stride_am + k * stride_an, lik)

# 3) Merge: result[i,j] = L[i,j] if j<=i else A_orig[i,j]  (strictly upper from original)
@triton.autotune(configs=generate_config_2d(), key=["N"], cache_results=True)
@triton.jit
def write_strict_upper_from_orig(out_ptr, stride_om, stride_on,
                                 A0_ptr,  stride_a0m, stride_a0n,
                                 N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    ii = pid_i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    jj = pid_j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mi = ii < N
    mj = jj < N
    m_ij = mi[:, None] & mj[None, :]

    # strictly upper mask: j > i
    m_upper = m_ij & (jj[None, :] > ii[:, None])

    # offsets
    off_out = ii[:, None] * stride_om + jj[None, :] * stride_on
    off_a0  = ii[:, None] * stride_a0m + jj[None, :] * stride_a0n

    # write A0's strictly upper into out; leave lower+diag (L) untouched
    vals = tl.load(A0_ptr + off_a0, mask=m_upper, other=0.0)
    tl.store(out_ptr + off_out, vals, mask=m_upper)

# ------------------------------------------------------
# Host-side function: drop-in for your numpy "kernel(A)"
# ------------------------------------------------------
def kernel(A: torch.Tensor):
    """
    In-place: A[:] = chol(A) + strictly_upper(original A)
    """
    N = A.shape[0]
    A0 = A.clone()  # keep original upper triangle

    stride_am, stride_an = A.stride()

    # Cholesky: overwrite A's lower triangle with L
    for k in range(N):
        # diag
        chol_diag_kernel[(1,)](A, stride_am, stride_an, N, k)
        # column below diag: launch one program per row i=k+1..N-1
        n_rows = max(0, N - (k + 1))
        if n_rows > 0:
            chol_col_kernel[(n_rows,)](A, stride_am, stride_an, N, k)

    # Merge: output = L (lower incl diag) + strictly upper from original A
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    write_strict_upper_from_orig[grid](
        A,   *A.stride(),
        A0,  *A0.stride(),
        N,
    )

    return A

