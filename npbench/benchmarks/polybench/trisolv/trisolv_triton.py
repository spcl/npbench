import torch
import triton
import triton.language as tl
import itertools
from npbench.infrastructure.triton_utilities import get_1d_tile_offsets

def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_K": k}, num_warps=w)
        for k, w in itertools.product(
            [512, 1024, 2048, 4096, 8192, 16384], [1, 2, 4, 8]
        )
    ]

@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def forward_subst_kernel(L, x, b, N: tl.constexpr, DTYPE : tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Loop over rows *sequentially*
    for i in range(0, N):
        # s = 0.0
        acc = tl.zeros((), dtype=DTYPE)

        # Parallelize the inner sum
        # for j in range(i):
        #    s += L[i, j] * x[j]
        for k0 in range(0, i, BLOCK_SIZE_K):
            j = k0 + tl.arange(0, BLOCK_SIZE_K)
            mask = j < i

            L_ij = tl.load(L + i * N + j, mask=mask, other=0.0)
            x_j  = tl.load(x + j,        mask=mask, other=0.0)

            acc += tl.sum(L_ij * x_j, axis=0)

        L_ii = tl.load(L + i * N + i)
        b_i  = tl.load(b + i)
        x_i  = (b_i - acc) / L_ii
        tl.store(x + i, x_i)

def kernel(L, x, b):
    # Assume A is a square matrix of size NxN
    N, M = L.shape
    x_len = x.shape[0]
    b_len = b.shape[0]
    assert x_len == N, "x length must match L dimensions"
    assert b_len == N, "b length must match L dimensions"
    assert N == M, "L must be a square matrix"

    dtype = L.dtype
    assert dtype in (torch.float32, torch.float64)

    DTYPE = tl.float32 if dtype == torch.float32 else tl.float64
    grid = (1,)  # one program instance for this system

    #     for i in range(N):
    #         s = 0.0
    #         for j in range(i):
    #             s += L[i, j] * x[j]
    #         x[i] = (b[i] - s) / L[i, i]
    # _kernel[grid](L, x, b, N, DTYPE)

    forward_subst_kernel[grid](L, x, b, N, DTYPE)
