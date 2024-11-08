import triton
import triton.language as tl
import itertools
import torch

# Automatically generate configurations based on block sizes
# Define possible block sizes to explore
block_sizes = [
    4,
    8,
    16,
    32,
    64,
    128
]
warp_counts = [1, 2, 4, 8, 16]
# Automatically generate configurations based on block sizes
configs = [
    triton.Config({"BLOCK_SIZE_X": x, "BLOCK_SIZE_Y": y}, num_warps=w)
    for x, y, w in list(itertools.product(block_sizes, block_sizes, warp_counts))
    if w * 32 <= x * y and x * y <= 1024 and (x * y) % (w * 32) == 0
]


@triton.autotune(configs=configs, key=["N"])
@triton.jit
def _kernel(
    TSTEPS,
    A_ptr: torch.Tensor,
    B_ptr: torch.Tensor,
    N,
    BLOCK_SIZE_X: triton.language.constexpr,
    BLOCK_SIZE_Y: triton.language.constexpr,
):
    # Calculate grid position
    pid_x = triton.language.program_id(0)
    pid_y = triton.language.program_id(1)
    grid_offset_x = pid_x * BLOCK_SIZE_X
    grid_offset_y = pid_y * BLOCK_SIZE_Y

    # Define the 2D tile indices
    i = grid_offset_x + triton.language.arange(0, BLOCK_SIZE_X)[:, None]
    j = grid_offset_y + triton.language.arange(0, BLOCK_SIZE_Y)[None, :]

    # Compute the offset for pointer arithmetic

    # Loop over time steps
    for t in range(TSTEPS):
        # Load values from A with boundary checks
        A_center = triton.language.load(
            A_ptr + (i + 1) * N + (j + 1), mask=(i < N - 2) & (j < N - 2), other=0.0
        )
        A_left = triton.language.load(
            A_ptr + (i + 1) * N + (j), mask=(i < N - 2) & (j < N - 2), other=0.0
        )
        A_right = triton.language.load(
            A_ptr + (i + 1) * N + (j + 2), mask=(i < N - 2) & (j < N - 2), other=0.0
        )
        A_top = triton.language.load(
            A_ptr + (i) * N + (j + 1), mask=(i < N - 2) & (j < N - 2), other=0.0
        )
        A_bottom = triton.language.load(
            A_ptr + (i + 2) * N + (j + 1), mask=(i < N - 2) & (j < N - 2), other=0.0
        )

        # Compute the new value for B at this position
        B_new = 0.20 * (A_left + A_right + A_top + A_bottom + A_center)
        triton.language.store(
            B_ptr + (i + 1) * N + (j + 1), B_new, mask=(i < N - 2) & (j < N - 2)
        )

        # Swap A and B pointers for the next timestep
        A_ptr, B_ptr = B_ptr, A_ptr


_jacobi_2d_triton_best_config = None

def autotuner(TSTEPS, A, B):
    global _jacobi_2d_triton_best_config
    M = int(A.shape[0])
    N = int(A.shape[1])
    assert N == M
    _kernel[(16, 16)](TSTEPS, A, B, M)
    _jacobi_2d_triton_best_config = _kernel.best_config

def kernel(TSTEPS, A, B):
    global _jacobi_2d_triton_best_config
    M = int(A.shape[0])
    N = int(A.shape[1])
    assert N == M
    assert _jacobi_2d_triton_best_config is not None
    best_config = _jacobi_2d_triton_best_config
    grid = (
        triton.cdiv(N, best_config.kwargs["BLOCK_SIZE_X"]),
        triton.cdiv(M, best_config.kwargs["BLOCK_SIZE_Y"]),
    )
    _kernel[grid](TSTEPS, A, B, M)
    torch.cuda.synchronize()
