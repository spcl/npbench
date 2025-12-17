import itertools
import torch
import triton
import triton.language as tl


def get_seidel_2d_configs():
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs, w in itertools.product(
            [64, 128, 256, 512, 1024],  # BLOCK_SIZE options
            [1, 2, 4, 8, 16, 32]        # num_warps options
        )
    ]


@triton.autotune(
    configs=get_seidel_2d_configs(),
    key=["N"],
    cache_results=True
)
@triton.jit
def _kernel_stencil(A_ptr, N, row_idx, BLOCK_SIZE: tl.constexpr):
  """Apply 7-neighbor stencil to row row_idx: A[row_idx, 1:-1] += (neighbors)"""
  # Parallelize across columns within this row
  col_block_id = tl.program_id(0)
  col_offsets = 1 + col_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  col_mask = col_offsets < (N - 1)

  # Base offset for row i
  row_base = row_idx * N

  prev_row_base = (row_idx - 1) * N
  top_left = tl.load(A_ptr + prev_row_base + (col_offsets - 1), mask=col_mask, other=0.0)
  top_center = tl.load(A_ptr + prev_row_base + col_offsets, mask=col_mask, other=0.0)
  top_right = tl.load(A_ptr + prev_row_base + (col_offsets + 1), mask=col_mask, other=0.0)

  # Row i (current)
  curr = tl.load(A_ptr + row_base + col_offsets, mask=col_mask, other=0.0)
  right = tl.load(A_ptr + row_base + (col_offsets + 1), mask=col_mask, other=0.0)

  # Row i+1 (below)
  next_row_base = (row_idx + 1) * N
  bottom_left = tl.load(A_ptr + next_row_base + (col_offsets - 1), mask=col_mask, other=0.0)
  bottom_center = tl.load(A_ptr + next_row_base + col_offsets, mask=col_mask, other=0.0)
  bottom_right = tl.load(A_ptr + next_row_base + (col_offsets + 1), mask=col_mask, other=0.0)

  # Apply stencil
  result = curr + top_left + top_center + top_right + right + bottom_left + bottom_center + bottom_right
  tl.store(A_ptr + row_base + col_offsets, result, mask=col_mask)


@triton.jit
def _kernel_recursive_scan(
    A_ptr,
    N,
):
    running_val = tl.load(A_ptr)
    for j in range(1, N - 1):
        ptr_curr = A_ptr + j
        curr_val = tl.load(ptr_curr)
        new_val = (curr_val + running_val) / 9.0
        tl.store(ptr_curr, new_val)
        running_val = new_val


def kernel(TMAX, N, A):
    grid_stencil = lambda meta: (triton.cdiv(N - 2, meta['BLOCK_SIZE']),)

    for t in range(TMAX - 1):
        # Process rows sequentially (Gauss-Seidel dependency)
        for i in range(1, N-1):
            # Apply stencil to row i in parallel across columns
            _kernel_stencil[grid_stencil](A, N, i)

            # Sequential scan along row i
            _kernel_recursive_scan[(1,)](
                A[i, :],
                N,
            )
