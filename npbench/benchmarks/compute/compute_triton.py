import torch
import triton
import triton.language as tl
import itertools
import numpy as np

def get_configs():
    return [
        triton.Config({"BLOCK_SIZE_N": block_size}, num_warps=num_warps)
        for block_size, num_warps in itertools.product(
            [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
    ]

@triton.autotune(configs=get_configs(), key=["N"], cache_results=True)
@triton.jit
def _kernel(array_1, array_2, a, b, c, N, arr_out, DTYPE: tl.constexpr,
            BLOCK_SIZE_N : tl.constexpr):

    #  def compute(array_1, array_2, a, b, c):
    #     return np.clip(array_1, 2, 10) * a + array_2 * b + c

    # clip(x) = 2 if x < 2
    # clip(x) = x if 2 <= x <= 10
    # clip(x) = 10 if x > 10

    pid_n = tl.program_id(axis=0)
    offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs < N

    arr1_vec = tl.load(array_1 + offs, mask=row_mask, other=0)
    arr2_vec = tl.load(array_2 + offs, mask=row_mask, other=0)

    # Clipping using masks: np.clip(array_1, 2, 10)
    mask_two = arr1_vec < 2 # true, true, false, ...
    mask_ten = arr1_vec > 10 # false, false, ... true

    two_vec = tl.full((BLOCK_SIZE_N,), 2, dtype=DTYPE)
    ten_vec = tl.full((BLOCK_SIZE_N,), 10, dtype=DTYPE)
    clipped_arr1 = tl.where(mask_two, two_vec, arr1_vec)
    clipped_arr1 = tl.where(mask_ten, ten_vec, clipped_arr1)

    # final computation 
    arr_out_vec = clipped_arr1 * a + arr2_vec * b + c

    tl.store(arr_out + offs, arr_out_vec, mask=row_mask)


def _as_py(x):
    # convert numpy scalar -> Python scalar
    return x.item() if isinstance(x, np.generic) else x


# expected the name of the kernel to be "compute" for some reason, error otherwise
def compute(array_1, array_2, a, b, c):
    # array_1, array_2: torch.int64, scalars a,b,c : numpy.int64
    N = array_1.numel()
    a2_len = array_2.numel()
    assert N == a2_len, "Input arrays must have the same length."

    # force type torch.float32 on arrays
    dtype = array_1.dtype
    if dtype not in (torch.int32, torch.int64):
        dtype = torch.int64
    DTYPE = tl.int64 if dtype == torch.int64 else tl.int32

    # Assume array_1, array_2, a, b, c have the same dtype
    # convert to dtype and make contiguous
    a1 = array_1.to(device= "cuda", dtype=dtype).contiguous()
    a2 = array_2.to(device= "cuda", dtype=dtype).contiguous()
    arr_out = torch.empty_like(a1)

    # kill numpy.* scalars -> python scalars
    a = int(_as_py(a))
    b = int(_as_py(b))
    c = int(_as_py(c))

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    _kernel[grid](a1, a2, a, b, c, N, arr_out, DTYPE)
    
    return arr_out



