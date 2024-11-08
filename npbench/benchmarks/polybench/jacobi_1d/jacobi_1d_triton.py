import triton
import torch
from npbench.infrastructure.triton_framework import TritonFramework

@triton.autotune(configs=TritonFramework.get_autotuner_configs_1D(), key=["n_elements"])
@triton.jit
def _kernel(TSTEPS, A, B, n_elements, BLOCK_SIZE_X: triton.language.constexpr):
    pid_x = triton.language.program_id(0)
    grid_offset_x = pid_x * BLOCK_SIZE_X
    i = grid_offset_x + triton.language.arange(0, BLOCK_SIZE_X)

    mask = (i > 0) & (i < n_elements - 2)

    for t in range(1, TSTEPS):
        A_left = triton.language.load(A + i, mask=mask, other=0.0)
        A_center = triton.language.load(A + i + 1, mask=mask, other=0.0)
        A_right = triton.language.load(A + i + 2, mask=mask, other=0.0)

        B_new = 0.33333 * (A_left + A_center + A_right)
        triton.language.store(B + i + 1, B_new, mask=mask)

        A, B = B, A


_jacobi_1d_triton_best_config = None


def autotuner(TSTEPS, A, B):
    global _jacobi_1d_triton_best_config
    M = int(A.shape[0])
    _kernel[(16,)](TSTEPS, A, B, M)
    _jacobi_1d_triton_best_config = _kernel.best_config


def kernel(TSTEPS, A, B):
    global _jacobi_1d_triton_best_config
    M = int(A.shape[0])
    assert _jacobi_1d_triton_best_config is not None
    best_config = _jacobi_1d_triton_best_config
    grid = (triton.cdiv(M, best_config.kwargs["BLOCK_SIZE_X"]),)
    _kernel[grid](TSTEPS, A, B, M)
    torch.cuda.synchronize()
