import itertools

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs, nw in itertools.product([64, 128, 256, 512], [1, 2, 4, 8])
    ],
    key=["N"],
    cache_results=True
)
@triton.jit
def durbin_kernel(
    y_ptr,
    y_temp_ptr,
    r_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    alpha = -tl.load(r_ptr)
    beta = alpha * 0.0 + 1.0
    tl.store(y_ptr, alpha)

    j_block = tl.arange(0, BLOCK_SIZE)
    for k in tl.range(1, N):
        beta = beta * (1.0 - alpha * alpha)
        r_k = tl.load(r_ptr + k)

        dot_product_sum = alpha * 0.0
        for j_start in tl.range(0, k, BLOCK_SIZE):
            j = j_start + j_block
            j_rev = (k - 1) - j

            mask = j < k
            j_rev_clamped = tl.where(mask, j_rev, 0)
            r_vec = tl.load(r_ptr + j_rev_clamped, mask=mask, other=0.0)
            y_vec = tl.load(y_ptr + j, mask=mask, other=0.0)
            dot_product_sum += tl.sum(r_vec * y_vec, axis=0)

        alpha = -(r_k + dot_product_sum) / beta

        # Copy y[:k] to temp buffer
        for j_start in tl.range(0, k, BLOCK_SIZE):
            j = j_start + j_block
            mask = j < k
            y_val = tl.load(y_ptr + j, mask=mask, other=0.0)
            tl.store(y_temp_ptr + j, y_val, mask=mask)

        tl.debug_barrier()

        # Update y[:k] using temp buffer
        for j_start in tl.range(0, k, BLOCK_SIZE):
            j = j_start + j_block
            j_rev = (k - 1) - j

            mask = j < k
            j_rev_clamped = tl.where(mask, j_rev, 0)

            y_old = tl.load(y_temp_ptr + j, mask=mask, other=0.0)
            y_rev_vec = tl.load(y_temp_ptr + j_rev_clamped, mask=mask, other=0.0)
            y_new = y_old + alpha * y_rev_vec
            tl.store(y_ptr + j, y_new, mask=mask)

        tl.store(y_ptr + k, alpha)


def kernel(r: torch.Tensor):
    N = r.shape[0]
    y = torch.empty_like(r)
    y_temp = torch.empty_like(r)

    durbin_kernel[(1,)](y, y_temp, r, N)
    return y
