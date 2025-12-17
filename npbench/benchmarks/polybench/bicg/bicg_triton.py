import itertools

import torch
import triton
import triton.language as tl
from npbench.infrastructure.triton_utilities import get_2d_tile_offsets

def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            [8, 16, 32, 64, 128], [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
        if m != 128 or n != 128
    ]

@triton.autotune(configs=generate_config(), key=["M", "N"], cache_results=True)
@triton.jit()
def _kernel(
        A, # (M, N)
        R, # (M, )
        P, # (N, )
        OUT0, # (M, )
        OUT1, # (N, )
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    tile, mask, row, column = get_2d_tile_offsets(
        x=j * BLOCK_SIZE_N,
        y=i * BLOCK_SIZE_M,
        tile_width=BLOCK_SIZE_N,
        tile_height=BLOCK_SIZE_M,
        matrix_width=N,
        matrix_height=M,
    )
    a = tl.load(A + tile, mask)
    r = tl.load(R + row, mask=row < M, other=0.0)
    p = tl.load(P + column, mask=column < N, other=0.0)

    r_sum = tl.sum(a * r[:, None], axis=0)
    p_sum = tl.sum(a * p[None, :], axis=1)
    tl.atomic_add(OUT0 + column, r_sum, sem="release")
    tl.atomic_add(OUT1 + row, p_sum, sem="release")


def kernel(A: torch.Tensor, p: torch.Tensor, r: torch.Tensor):
    # return r @ A, A @ p
    out0 = torch.zeros((A.shape[1],), dtype=A.dtype)
    out1 = torch.zeros((A.shape[0],), dtype=A.dtype)

    grid = lambda meta: (
        triton.cdiv(A.shape[0], meta["BLOCK_SIZE_M"]),
        triton.cdiv(A.shape[1], meta["BLOCK_SIZE_N"]),
    )
    _kernel[grid](A, r, p, out0, out1, A.shape[0], A.shape[1])
    return out0, out1
