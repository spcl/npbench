import itertools
import torch
import triton
import triton.language as tl
from npbench.infrastructure.triton_utilities import matmul


"""
SOLUTION 1

Computes (A@B)@C and then in a grid it computes alpha*R + beta*D

python3 run_benchmark.py -b k2mm -f triton -p paper -v True
***** Testing Triton with k2mm on the paper dataset, datatype default *****
NumPy - default - validation: 1127ms
Triton - default - first/validation: 39652ms
Triton - default - default - validation: SUCCESS
Triton - default - median: 8472ms
"""
# def generate_config():
#     return [
#         triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
#         for m, n, w in itertools.product(
#             [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
#         )
#         if m != 128 or n != 128
#     ]

# @triton.autotune(configs=generate_config(), key=["M", "N"], cache_results=True)
# @triton.jit
# def _kernel(
#     R_ptr, D_ptr,
#     M: tl.int32, N: tl.int32,
#     stride_rm: tl.int32, stride_rn: tl.int32,
#     stride_dm: tl.int32, stride_dn: tl.int32,
#     alpha, beta,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

#     mask_m = offs_m < M
#     mask_n = offs_n < N
#     mask = mask_m[:, None] & mask_n[None, :]

#     r_ptrs = R_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
#     d_ptrs = D_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn

#     r = tl.load(r_ptrs, mask=mask)
#     d = tl.load(d_ptrs, mask=mask)

#     out = alpha * r + beta * d
#     tl.store(d_ptrs, out, mask=mask)

# def kernel(alpha: float, beta: float, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
#     T = matmul(A, B)
#     res = matmul(T, C)

#     M, N = D.shape
    
#     grid = lambda meta: (
#         triton.cdiv(M, meta["BLOCK_SIZE_M"]),
#         triton.cdiv(N, meta["BLOCK_SIZE_N"]),
#     )

#     _kernel[grid](
#         res, D,
#         M, N,
#         res.stride(0), res.stride(1),
#         D.stride(0), D.stride(1),
#         alpha, beta,
#     )

"""
SOLUTION 2

Same as previous, but instead solve it in a one-dimension

python3 run_benchmark.py -b k2mm -f triton -p paper -v True
***** Testing Triton with k2mm on the paper dataset, datatype default *****
NumPy - default - validation: 1115ms
Triton - default - first/validation: 14239ms
Triton - default - default - validation: SUCCESS
Triton - default - median: 8472ms
"""
def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE": m}, num_warps=w)
        for m, w in itertools.product(
            [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
    ]

@triton.autotune(configs=generate_config(), key=["size"], cache_results=True)
@triton.jit
def _kernel(alpha: float, beta: float, RES: torch.Tensor, D: torch.Tensor, size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    r = tl.load(RES + offsets, mask=mask)
    d = tl.load(D + offsets, mask=mask)

    out = alpha * r + beta * d
    tl.store(D + offsets, out, mask=mask)

def kernel(alpha: float, beta: float, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
    T = matmul(A, B)
    res = matmul(T, C)
    
    size = D.numel()
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

    _kernel[grid](alpha, beta, res, D, size)


"""
SOLUTION 3

First compute (A@B) or (B@C) (depending which one is smaller),
and then while calculating T@R calculate alpha*R + beta*D immediately


python3 run_benchmark.py -b k2mm -f triton -p paper -v True
***** Testing Triton with k2mm on the paper dataset, datatype default *****
NumPy - default - validation: 1681ms
Triton - default - first/validation: 20248ms
Triton - default - default - validation: SUCCESS
Triton - default - median: 20007ms
"""
# @triton.jit
# def mma_dot(a, b):
#     return tl.dot(a, b)

# @triton.jit
# def mma_outer(a, b):
#     return tl.sum(a[:, :, None] * b[None, :, :], axis=1)

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=2),
#     ],
#     key=["M", "N", "K"]
# )
# @triton.jit
# def _gemm_epilogue_kernel(
#     A_ptr, B_ptr, D_ptr,
#     M: tl.int32, N: tl.int32, K: tl.int32,
#     stride_am: tl.int32, stride_ak: tl.int32,  # A: (M, K)
#     stride_bk: tl.int32, stride_bn: tl.int32,  # B: (K, N)
#     stride_dm: tl.int32, stride_dn: tl.int32,  # D: (M, N)
#     alpha, beta,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     MATRIX_MULT: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)         # (BM,)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)         # (BN,)

#     d_ptrs = D_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
#     mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)

#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=D_ptr.dtype.element_ty)

#     for k0 in range(0, K, BLOCK_SIZE_K):
#         k_ids = k0 + tl.arange(0, BLOCK_SIZE_K)                   # (BK,)
#         a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak  # (BM, BK)
#         b_ptrs = B_ptr + k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn  # (BK, BN)

#         a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0)
#         b = tl.load(b_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0)

#         acc += MATRIX_MULT(a, b)

#     # Fused epilogue: out = alpha * acc + beta * D
#     d_old = tl.load(d_ptrs, mask=mask_mn, other=0)
#     out = alpha * acc + beta * d_old

#     tl.store(d_ptrs, out, mask=mask_mn)

# def _gemm_epilogue(alpha, beta, A: torch.Tensor, B: torch.Tensor, D: torch.Tensor):
#     """
#     Compute D <- alpha * (A @ B) + beta * D directly with a fused epilogue
#     A: (M, K), B: (K, N), D: (M, N)
#     """
#     M, K = A.shape
#     _, N = B.shape

#     grid = lambda META: (
#         triton.cdiv(M, META["BLOCK_SIZE_M"]),
#         triton.cdiv(N, META["BLOCK_SIZE_N"]),
#     )

#     MMA = mma_dot if A.dtype is torch.float32 else mma_outer

#     _gemm_epilogue_kernel[grid](
#         A, B, D,
#         M, N, K,
#         A.stride(0), A.stride(1),
#         B.stride(0), B.stride(1),
#         D.stride(0), D.stride(1),
#         alpha, beta,
#         MATRIX_MULT=MMA,
#     )


# def kernel(alpha: float, beta: float,
#            A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
#     """
#     Compute: D[:] = alpha * (A @ B @ C) + beta * D

#       1) Build R1 with the cheaper association:
#            - if M*K2 <= K1*N: R1 = A @ B  (shape M x K2), then fused GEMM/epilogue with C
#            - else:            R1 = B @ C  (shape K1 x N), then fused GEMM/epilogue with A
#       2) The second GEMM writes directly into D with the epilogue fused:
#            D <- alpha * (second GEMM) + beta * D
#     """
#     M, K1 = A.shape
#     K2, N = C.shape

#     cost1 = M * K2      # intermediate if R1 = A@B
#     cost2 = K1 * N      # intermediate if R1 = B@C

#     if cost1 <= cost2:
#         # R1 = A @ B
#         R1 = matmul(A, B)
#         # D = alpha * (R1 @ C) + beta * D
#         _gemm_epilogue(alpha, beta, R1, C, D)
#     else:
#         # R1 = B @ C
#         R1 = matmul(B, C)
#         # D = alpha * (A @ R1) + beta * D
#         _gemm_epilogue(alpha, beta, A, R1, D)

"""
SOLUTION 4

This one performs all multiplication inside a kernel to avoid multiple memory round trips

Didn't finish lol
"""
# @triton.jit
# def mma_dot(a, b):
#     return tl.dot(a, b)

# @triton.jit
# def mma_outer(a, b):
#     return tl.sum(a[:, :, None] * b[None, :, :], axis=1)

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K1': 64, 'BLOCK_SIZE_K2': 64}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K1': 64, 'BLOCK_SIZE_K2': 32}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K1': 64, 'BLOCK_SIZE_K2': 32}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K1': 64, 'BLOCK_SIZE_K2': 64}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K1': 32, 'BLOCK_SIZE_K2': 32}, num_warps=8),
#     ],
#     key=["M", "N", "K1", "K2"]
# )
# @triton.jit
# def _abc_epilogue_kernel(
#     A_ptr, B_ptr, C_ptr, D_ptr,
#     M: tl.int32, N: tl.int32, K1: tl.int32, K2: tl.int32,
#     # A: (M, K1)
#     stride_am: tl.int32, stride_ak1: tl.int32,
#     # B: (K1, K2)
#     stride_bk1: tl.int32, stride_bk2: tl.int32,
#     # C: (K2, N)
#     stride_ck2: tl.int32, stride_cn: tl.int32,
#     # D: (M, N)
#     stride_dm: tl.int32, stride_dn: tl.int32,
#     alpha, beta,
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K1: tl.constexpr, BLOCK_SIZE_K2: tl.constexpr,
#     MATRIX_MULT: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)

#     offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # (BM,)
#     offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # (BN,)

#     # Output pointers + bounds mask
#     d_ptrs = D_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
#     mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)

#     # Accumulator for the final output tile (BM x BN)
#     acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=D_ptr.dtype.element_ty)

#     for k2_0 in range(0, K2, BLOCK_SIZE_K2):
#         offs_k2 = k2_0 + tl.arange(0, BLOCK_SIZE_K2)  # (BK2,)

#         # Temporary tile T = (A @ B[:, k2_0:k2_0+BK2])  -> shape (BM x BK2)
#         T = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K2), dtype=D_ptr.dtype.element_ty)

#         for k1_0 in range(0, K1, BLOCK_SIZE_K1):
#             offs_k1 = k1_0 + tl.arange(0, BLOCK_SIZE_K1)  # (BK1,)

#             # A_tile: (BM, BK1)
#             a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k1[None, :] * stride_ak1
#             a_mask = (offs_m[:, None] < M) & (offs_k1[None, :] < K1)
#             a = tl.load(a_ptrs, mask=a_mask, other=0)

#             # B_tile: (BK1, BK2)
#             b_ptrs = B_ptr + offs_k1[:, None] * stride_bk1 + offs_k2[None, :] * stride_bk2
#             b_mask = (offs_k1[:, None] < K1) & (offs_k2[None, :] < K2)
#             b = tl.load(b_ptrs, mask=b_mask, other=0)

#             T += MATRIX_MULT(a, b)  # (BM x BK2)

#         # Multiply the partial T with C_tile and accumulate into acc
#         # C_tile: (BK2, BN)
#         c_ptrs = C_ptr + offs_k2[:, None] * stride_ck2 + offs_n[None, :] * stride_cn
#         c_mask = (offs_k2[:, None] < K2) & (offs_n[None, :] < N)
#         c = tl.load(c_ptrs, mask=c_mask, other=0)

#         acc += MATRIX_MULT(T, c)  # (BM x BN)

#     # Fused epilogue: D = alpha * acc + beta * D
#     d_old = tl.load(d_ptrs, mask=mask_mn, other=0)
#     out = alpha * acc + beta * d_old
#     tl.store(d_ptrs, out, mask=mask_mn)


# def kernel(alpha: float, beta: float,
#            A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
#     """
#     Compute: D <- alpha * (A @ B @ C) + beta * D
#     in a single kernel without writing intermediates to global memory.
#     Shapes:
#       A: (M, K1), B: (K1, K2), C: (K2, N), D: (M, N)
#     """

#     M, K1 = A.shape
#     K1b, K2 = B.shape
#     K2c, N = C.shape

#     # Launch grid over output tiles (M, N)
#     grid = lambda META: (
#         triton.cdiv(M, META["BLOCK_SIZE_M"]),
#         triton.cdiv(N, META["BLOCK_SIZE_N"]),
#     )

#     # Choose matmul micro-op depending on dtype (you can refine this)
#     MMA = mma_dot if A.dtype in (torch.float16, torch.bfloat16, torch.float32) else mma_outer

#     _abc_epilogue_kernel[grid](
#         A, B, C, D,
#         M, N, K1, K2,
#         A.stride(0), A.stride(1),       # A: (M, K1)
#         B.stride(0), B.stride(1),       # B: (K1, K2)
#         C.stride(0), C.stride(1),       # C: (K2, N)
#         D.stride(0), D.stride(1),       # D: (M, N)
#         float(alpha), float(beta),
#         MATRIX_MULT=MMA,
#     )
