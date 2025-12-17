import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw)
        for nw in [1, 2, 4, 8]
    ],
    key=["I", "J", "K"],
    cache_results=True
)
@triton.jit
def vadv_kernel(
    utens_stage_ptr,
    u_stage_ptr,
    wcon_ptr,
    u_pos_ptr,
    utens_ptr,
    ccol_ptr,
    dcol_ptr,
    data_col_ptr,
    dtr_stage,
    I, J, K,
):
    ij_idx = tl.program_id(0)
    i = ij_idx // J
    j = ij_idx % J

    if i >= I or j >= J:
        return

    wcon_i = i + 1

    k = 0
    wcon_k1_0 = tl.load(wcon_ptr + wcon_i * J * K + j * K + k + 1)
    wcon_k1_m1 = tl.load(wcon_ptr + (wcon_i - 1) * J * K + j * K + k + 1)
    gcv = 0.25 * (wcon_k1_0 + wcon_k1_m1)
    cs = gcv * 0.5

    ccol_val = gcv * 0.5
    tl.store(ccol_ptr + i * J * K + j * K + k, ccol_val)
    bcol = dtr_stage - ccol_val

    u_stage_k = tl.load(u_stage_ptr + i * J * K + j * K + k)
    u_stage_k1 = tl.load(u_stage_ptr + i * J * K + j * K + k + 1)
    correction_term = -cs * (u_stage_k1 - u_stage_k)

    u_pos_k = tl.load(u_pos_ptr + i * J * K + j * K + k)
    utens_k = tl.load(utens_ptr + i * J * K + j * K + k)
    utens_stage_k = tl.load(utens_stage_ptr + i * J * K + j * K + k)
    dcol_val = dtr_stage * u_pos_k + utens_k + utens_stage_k + correction_term

    divided = 1.0 / bcol
    ccol_val = ccol_val * divided
    dcol_val = dcol_val * divided
    tl.store(ccol_ptr + i * J * K + j * K + k, ccol_val)
    tl.store(dcol_ptr + i * J * K + j * K + k, dcol_val)

    for k in tl.range(1, K - 1):
        wcon_k_0 = tl.load(wcon_ptr + wcon_i * J * K + j * K + k)
        wcon_k_m1 = tl.load(wcon_ptr + (wcon_i - 1) * J * K + j * K + k)
        gav = -0.25 * (wcon_k_0 + wcon_k_m1)

        wcon_k1_0 = tl.load(wcon_ptr + wcon_i * J * K + j * K + k + 1)
        wcon_k1_m1 = tl.load(wcon_ptr + (wcon_i - 1) * J * K + j * K + k + 1)
        gcv = 0.25 * (wcon_k1_0 + wcon_k1_m1)

        as_ = gav * 0.5
        cs = gcv * 0.5

        acol = gav * 0.5
        ccol_val = gcv * 0.5
        bcol = dtr_stage - acol - ccol_val

        u_stage_km1 = tl.load(u_stage_ptr + i * J * K + j * K + k - 1)
        u_stage_k = tl.load(u_stage_ptr + i * J * K + j * K + k)
        u_stage_k1 = tl.load(u_stage_ptr + i * J * K + j * K + k + 1)
        correction_term = -as_ * (u_stage_km1 - u_stage_k) - cs * (u_stage_k1 - u_stage_k)

        u_pos_k = tl.load(u_pos_ptr + i * J * K + j * K + k)
        utens_k = tl.load(utens_ptr + i * J * K + j * K + k)
        utens_stage_k = tl.load(utens_stage_ptr + i * J * K + j * K + k)
        dcol_val = dtr_stage * u_pos_k + utens_k + utens_stage_k + correction_term

        ccol_km1 = tl.load(ccol_ptr + i * J * K + j * K + k - 1)
        divided = 1.0 / (bcol - ccol_km1 * acol)
        ccol_val = ccol_val * divided

        dcol_km1 = tl.load(dcol_ptr + i * J * K + j * K + k - 1)
        dcol_val = (dcol_val - dcol_km1 * acol) * divided

        tl.store(ccol_ptr + i * J * K + j * K + k, ccol_val)
        tl.store(dcol_ptr + i * J * K + j * K + k, dcol_val)

    k = K - 1
    wcon_k_0 = tl.load(wcon_ptr + wcon_i * J * K + j * K + k)
    wcon_k_m1 = tl.load(wcon_ptr + (wcon_i - 1) * J * K + j * K + k)
    gav = -0.25 * (wcon_k_0 + wcon_k_m1)
    as_ = gav * 0.5
    acol = gav * 0.5
    bcol = dtr_stage - acol

    u_stage_km1 = tl.load(u_stage_ptr + i * J * K + j * K + k - 1)
    u_stage_k = tl.load(u_stage_ptr + i * J * K + j * K + k)
    correction_term = -as_ * (u_stage_km1 - u_stage_k)

    u_pos_k = tl.load(u_pos_ptr + i * J * K + j * K + k)
    utens_k = tl.load(utens_ptr + i * J * K + j * K + k)
    utens_stage_k = tl.load(utens_stage_ptr + i * J * K + j * K + k)
    dcol_val = dtr_stage * u_pos_k + utens_k + utens_stage_k + correction_term

    ccol_km1 = tl.load(ccol_ptr + i * J * K + j * K + k - 1)
    dcol_km1 = tl.load(dcol_ptr + i * J * K + j * K + k - 1)
    divided = 1.0 / (bcol - ccol_km1 * acol)
    dcol_val = (dcol_val - dcol_km1 * acol) * divided
    tl.store(dcol_ptr + i * J * K + j * K + k, dcol_val)

    k = K - 1
    datacol = tl.load(dcol_ptr + i * J * K + j * K + k)
    tl.store(data_col_ptr + i * J + j, datacol)
    u_pos_k = tl.load(u_pos_ptr + i * J * K + j * K + k)
    tl.store(utens_stage_ptr + i * J * K + j * K + k, dtr_stage * (datacol - u_pos_k))

    for k in tl.range(K - 2, -1, -1):
        ccol_k = tl.load(ccol_ptr + i * J * K + j * K + k)
        data_col_val = tl.load(data_col_ptr + i * J + j)
        dcol_k = tl.load(dcol_ptr + i * J * K + j * K + k)
        datacol = dcol_k - ccol_k * data_col_val
        tl.store(data_col_ptr + i * J + j, datacol)
        u_pos_k = tl.load(u_pos_ptr + i * J * K + j * K + k)
        tl.store(utens_stage_ptr + i * J * K + j * K + k, dtr_stage * (datacol - u_pos_k))


def vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    I, J, K = utens_stage.shape

    ccol = torch.empty_like(utens_stage)
    dcol = torch.empty_like(utens_stage)
    data_col = torch.empty((I, J), dtype=utens_stage.dtype, device=utens_stage.device)

    grid = (I * J,)
    vadv_kernel[grid](
        utens_stage, u_stage, wcon, u_pos, utens,
        ccol, dcol, data_col,
        float(dtr_stage),
        I, J, K
    )
