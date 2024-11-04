import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['N']
)
@triton.jit
def stencil_kernel(A_ptr, B_ptr, N, TSTEPS, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
    pid = tl.program_id(0)
    grid_offset = pid * BLOCK_SIZE

    i = grid_offset + tl.arange(0, BLOCK_SIZE)
    j = tl.arange(0, BLOCK_SIZE)

    for t in range(1, TSTEPS):
        A_center = tl.load(A_ptr + i * N + j, mask=(i < N) & (j < N), other=0.0)
        A_top = tl.load(A_ptr + (i - 1) * N + j, mask=(i > 0) & (j < N), other=0.0)
        A_bottom = tl.load(A_ptr + (i + 1) * N + j, mask=(i < N - 1) & (j < N), other=0.0)
        A_left = tl.load(A_ptr + i * N + (j - 1), mask=(i < N) & (j > 0), other=0.0)
        A_right = tl.load(A_ptr + i * N + (j + 1), mask=(i < N) & (j < N - 1), other=0.0)

        B_new = 0.2 * (A_center + A_top + A_bottom + A_left + A_right)
        tl.store(B_ptr + i * N + j, B_new, mask=(i < N) & (j < N))

        tl.synchronize()

        B_center = tl.load(B_ptr + i * N + j, mask=(i < N) & (j < N), other=0.0)
        B_top = tl.load(B_ptr + (i - 1) * N + j, mask=(i > 0) & (j < N), other=0.0)
        B_bottom = tl.load(B_ptr + (i + 1) * N + j, mask=(i < N - 1) & (j < N), other=0.0)
        B_left = tl.load(B_ptr + i * N + (j - 1), mask=(i < N) & (j > 0), other=0.0)
        B_right = tl.load(B_ptr + i * N + (j + 1), mask=(i < N) & (j < N - 1), other=0.0)

        A_new = 0.2 * (B_center + B_top + B_bottom + B_left + B_right)
        tl.store(A_ptr + i * N + j, A_new, mask=(i < N) & (j < N))