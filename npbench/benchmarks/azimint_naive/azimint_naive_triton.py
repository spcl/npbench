import itertools
import torch
import triton
import triton.language as tl

def generate_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE": m}, num_warps=w)
        for m, w in itertools.product(
            [256, 512, 1024], [1, 2, 4, 8]
        )
    ]


def generate_config_npt():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_NPT": m}, num_warps=w)
        for m, w in itertools.product(
            [8, 16, 32, 64, 128], [1, 2, 4, 8]
        )
    ]

@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def _kernel_max(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = tl.max(x, axis=0)
    tl.store(out_ptr + pid, m)


def triton_max(x: torch.Tensor):
    cur = x
    n = cur.numel()

    MIN_BLOCK = 8
    while n > 1:
        grid_size = triton.cdiv(n, MIN_BLOCK)
        out = torch.empty(grid_size, dtype=cur.dtype)
        _kernel_max[(grid_size,)](cur, out, n)
        cur = out
        n = cur.numel()
    return cur[0]

@triton.autotune(configs=generate_config(), key=["N"], cache_results=True)
@triton.jit
def _accumulate_bins_kernel(data_ptr, radius_ptr,
                            sums_ptr, counts_ptr,
                            N, n_bins, rmax: tl.float64,
                            BLOCK_SIZE: tl.constexpr):
    # axis 0 = bin index; axis 1 = block id over the data
    bin_idx = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    r = tl.load(radius_ptr + offs, mask=mask, other=0.0)
    rmax64 = tl.full((), rmax, tl.float64)
    n_bins64 = tl.full((), n_bins, tl.float64)
    bin64 = bin_idx.to(tl.float64)

    r1 = rmax64 * bin64 / n_bins64
    r2 = rmax64 * (bin64 + 1.0) / n_bins64

    # faster version but worse error:
    # r1 = rmax * bin_idx / n_bins
    # r2 = rmax * (bin_idx + 1.0) / n_bins

    in_bin = (r1 <= r) & (r < r2)

    v = tl.load(data_ptr + offs, mask=mask & in_bin, other=0.0)
    value = tl.sum(v, axis=0)

    counter = tl.sum((in_bin & mask), axis=0)

    tl.atomic_add(sums_ptr + bin_idx, value)
    tl.atomic_add(counts_ptr + bin_idx, counter)

@triton.autotune(configs=generate_config_npt(), key=["n_bins"], cache_results=True)
@triton.jit
def _finalize_means_kernel(sums_ptr, counts_ptr, means_ptr, n_bins,
                           BLOCK_SIZE_NPT: tl.constexpr):
    pid  = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE_NPT + tl.arange(0, BLOCK_SIZE_NPT)
    mask = offs < n_bins

    s = tl.load(sums_ptr   + offs, mask=mask, other=0.0)
    c = tl.load(counts_ptr + offs, mask=mask, other=0)

    mean = tl.where(c > 0, s / c, 0.0)
    tl.store(means_ptr + offs, mean, mask=mask)


def azimint_naive(data: torch.Tensor, radius: torch.Tensor, npt: int):
    N = data.numel()

    rmax = triton_max(radius).item()

    sums   = torch.zeros(npt, dtype=torch.float64)
    counts = torch.zeros(npt, dtype=torch.int32)
    means  = torch.empty(npt, dtype=torch.float64)

    grid = lambda meta: (npt, triton.cdiv(N, meta["BLOCK_SIZE"]),)
    _accumulate_bins_kernel[grid](
        data, radius, sums, counts, N, npt, rmax,
    )

    grid = lambda meta: (triton.cdiv(npt, meta["BLOCK_SIZE_NPT"]),)
    _finalize_means_kernel[grid](
        sums, counts, means, npt,
    )
    return means