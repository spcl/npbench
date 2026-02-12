import itertools
import torch
import triton
import triton.language as tl

def generate_config():
    ms = [64, 128]
    ns = [64, 128]
    ks = [32, 64]
    cfgs = []
    for m, n, k in itertools.product(ms, ns, ks):
        if m == 128 and n == 128 and k == 64:
            pass
        cfgs.append(
            triton.Config(
                kwargs={
                    "BLOCK_SIZE_M": m,
                    "BLOCK_SIZE_N": n,
                    "BLOCK_SIZE_K": k,
                },
            )
        )
    return cfgs

@triton.jit
def mma_dot(a, b):
    return tl.dot(a, b)

@triton.jit
def mma_outer(a, b):
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)

@triton.autotune(configs=generate_config(), key=["M", "N"], cache_results=True)
@triton.jit
def _symm_lower_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MATRIX_MULT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_in = offs_m[:, None] < M            # (BM, 1)
    n_in = offs_n[None, :] < N            # (1, BN)

    # Pointers to block of C and its mask
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_in & n_in                  # (BM, BN)

    # Accumulator for S@B
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=A_ptr.dtype.element_ty)

    # Loop over K tiles
    for k0 in range(0, M, BLOCK_SIZE_K):
        k = k0 + offs_k                   # [BK]
        k_in_row = k[None, :] < M         # (1, BK)
        k_in_col = k[:, None] < M         # (BK, 1)

        # We need S[m, k]: if m >= k -> A[m,k]; else -> A[k,m]
        a_ptrs_lower = A_ptr + (offs_m[:, None] * stride_am + k[None, :] * stride_ak)   # (BM, BK)
        a_ptrs_upper = A_ptr + (k[None, :] * stride_am + offs_m[:, None] * stride_ak)   # (BM, BK)

        m_idx = tl.broadcast_to(offs_m[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        k_idx = tl.broadcast_to(k[None, :],      (BLOCK_SIZE_M, BLOCK_SIZE_K))
        use_lower = m_idx >= k_idx                                                   # (BM, BK)

        # masks shaped like (BM, BK)
        ak_mask = (m_in & k_in_row)                                                  # (BM, BK)
        a_lower = tl.load(a_ptrs_lower, mask=(ak_mask & use_lower), other=0.0)
        a_upper = tl.load(a_ptrs_upper, mask=(ak_mask & ~use_lower), other=0.0)
        s_tile = a_lower + a_upper                                                   # (BM, BK)

        # Load B[k, n] -> (BK, BN)
        b_ptrs = B_ptr + (k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        bn_mask = (k_in_col & n_in)                                                  # (BK, BN)
        b_tile = tl.load(b_ptrs, mask=bn_mask, other=0.0)

        acc += MATRIX_MULT(s_tile, b_tile)

    # Scale and write back: C = beta*C + alpha*acc
    c_old = tl.load(c_ptrs, mask=c_mask, other=0.0)
    c_new = beta * c_old + alpha * acc
    tl.store(c_ptrs, c_new, mask=c_mask)


def symm_lower_mm(alpha, beta, A, B, C):
    """
    Compute C = beta*C + alpha * (S @ B), where S is the symmetric matrix
    formed from the lower triangle of A (A's upper triangle is ignored).
    """
    M, N = B.shape

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    MMA = mma_dot if A.dtype in (torch.float16, torch.bfloat16, torch.float32) else mma_outer

    _symm_lower_mm_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        float(alpha), float(beta),
        MATRIX_MULT=MMA,
    )
    return C


def kernel(alpha, beta, C: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    return symm_lower_mm(alpha, beta, A, B, C)
