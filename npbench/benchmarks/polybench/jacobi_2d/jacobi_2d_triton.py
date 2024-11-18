import triton
import torch
from npbench.infrastructure.triton_framework import TritonFramework


@triton.autotune(configs=TritonFramework.get_autotuner_configs_2D(), key=["N"])
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

_jacobi_2d_triton_best_config = None

def autotuner(TSTEPS, A, B):
    global _jacobi_2d_triton_best_config

    if _jacobi_2d_triton_best_config is not None:
        return

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
    for _ in range(1, TSTEPS):
        _kernel[grid](TSTEPS, A, B, M)
        _kernel[grid](TSTEPS, B, A, M)
    return A