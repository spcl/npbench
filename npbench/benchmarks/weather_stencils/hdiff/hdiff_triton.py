import torch
import triton
import triton.language as tl
import itertools

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': b}, num_warps=w)
        for b, w in itertools.product([8, 16, 32, 64, 128, 256], [1, 2, 4, 8])
    ],
    key=['I', 'J', 'K'],
    cache_results=True
)
@triton.jit
def hdiff_kernel(
    in_field_ptr,
    out_field_ptr,
    coeff_ptr,
    I: tl.int32,
    J: tl.int32,
    K: tl.int32,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for horizontal diffusion, fusing all intermediate steps.
    
    This kernel calculates the output for one (i, j) column, processing
    BLOCK_SIZE_K elements in the k-dimension at a time.
    """
    i = tl.program_id(0)
    j = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < K

    # Load 5x5 Input Patch
    # To compute out[i, j, k], we need a 5x5 patch from in_field,
    # starting at in[i, j, k].
    # We load all 13 necessary values for the k-block.
    
    # Pre-calculate base pointers for the (i, j) location
    stride_in_i = K * (J + 4)
    in_ptr = in_field_ptr + i * stride_in_i + j * K
    
    # Load the 5x5 patch (13 loads)
    # Row i
    in_i0_j2 = tl.load(
        in_ptr + 0 * stride_in_i + 2 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    
    # Row i+1
    in_i1_j1 = tl.load(
        in_ptr + 1 * stride_in_i + 1 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i1_j2 = tl.load(
        in_ptr + 1 * stride_in_i + 2 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i1_j3 = tl.load(
        in_ptr + 1 * stride_in_i + 3 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    
    # Row i+2
    in_i2_j0 = tl.load(
        in_ptr + 2 * stride_in_i + 0 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i2_j1 = tl.load(
        in_ptr + 2 * stride_in_i + 1 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i2_j2 = tl.load( # This is the "center"
        in_ptr + 2 * stride_in_i + 2 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i2_j3 = tl.load(
        in_ptr + 2 * stride_in_i + 3 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i2_j4 = tl.load(
        in_ptr + 2 * stride_in_i + 4 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    
    # Row i+3
    in_i3_j1 = tl.load(
        in_ptr + 3 * stride_in_i + 1 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i3_j2 = tl.load(
        in_ptr + 3 * stride_in_i + 2 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    in_i3_j3 = tl.load(
        in_ptr + 3 * stride_in_i + 3 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    
    # Row i+4
    in_i4_j2 = tl.load(
        in_ptr + 4 * stride_in_i + 2 * K + k_offsets,
        mask=k_mask, other=0.0
    )
    
    # --- 4. Load Coefficient ---
    coeff = tl.load(
        coeff_ptr + i * J * K + j * K + k_offsets,
        mask=k_mask,
        other=0.0
    )
    # Naming: lap_i_j1 corresponds to lap_field[i, j+1, k]
    
    # lap_field[i, j+1, k]
    lap_i_j1 = 4.0 * in_i1_j2 - (in_i2_j2 + in_i0_j2 + in_i1_j3 + in_i1_j1)
    
    # lap_field[i+1, j, k]
    lap_i1_j = 4.0 * in_i2_j1 - (in_i3_j1 + in_i1_j1 + in_i2_j2 + in_i2_j0)
    
    # lap_field[i+1, j+1, k]
    lap_i1_j1 = 4.0 * in_i2_j2 - (in_i3_j2 + in_i1_j2 + in_i2_j3 + in_i2_j1)
    
    # lap_field[i+1, j+2, k]
    lap_i1_j2 = 4.0 * in_i2_j3 - (in_i3_j3 + in_i1_j3 + in_i2_j4 + in_i2_j2)
    
    # lap_field[i+2, j+1, k]
    lap_i2_j1 = 4.0 * in_i3_j2 - (in_i4_j2 + in_i2_j2 + in_i3_j3 + in_i3_j1)

    # flx_field[i, j, k]
    res_flx_i = lap_i1_j1 - lap_i_j1
    cond_flx_i = in_i2_j2 - in_i1_j2
    flx_i = tl.where((res_flx_i * cond_flx_i) > 0.0, 0.0, res_flx_i)
    
    # flx_field[i+1, j, k]
    res_flx_i1 = lap_i2_j1 - lap_i1_j1
    cond_flx_i1 = in_i3_j2 - in_i2_j2
    flx_i1 = tl.where((res_flx_i1 * cond_flx_i1) > 0.0, 0.0, res_flx_i1)
    
    # fly_field[i, j, k]
    res_fly_j = lap_i1_j1 - lap_i1_j
    cond_fly_j = in_i2_j2 - in_i2_j1
    fly_j = tl.where((res_fly_j * cond_fly_j) > 0.0, 0.0, res_fly_j)
    
    # fly_field[i, j+1, k]
    res_fly_j1 = lap_i1_j2 - lap_i1_j1
    cond_fly_j1 = in_i2_j3 - in_i2_j2
    fly_j1 = tl.where((res_fly_j1 * cond_fly_j1) > 0.0, 0.0, res_fly_j1)
    
    # Divergence term
    flx_div = flx_i1 - flx_i
    fly_div = fly_j1 - fly_j
    div = flx_div + fly_div
    
    out = in_i2_j2 - coeff * div

    out_ptr = out_field_ptr + i * J * K + j * K + k_offsets
    tl.store(out_ptr, out, mask=k_mask)


def hdiff(in_field: torch.Tensor, out_field: torch.Tensor, coeff: torch.Tensor):
    I, J, K = out_field.shape
    
    grid = lambda meta: (I, J, triton.cdiv(K, meta['BLOCK_SIZE_K']))
    hdiff_kernel[grid](
        in_field, out_field, coeff,
        I, J, K,
    )
    
    return out_field