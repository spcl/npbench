import itertools

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.language.extra import libdevice

from npbench.infrastructure.triton_framework import tl_float
from npbench.infrastructure.triton_utilities import derive_launch_arguments, use_grid, complex_div, complex_mul, \
    powers_of_2, get_2d_tile_offsets


def generate_config_2d():
    if knobs.runtime.interpret:
        return [triton.Config(kwargs={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 2})]
    return [
        triton.Config(kwargs={"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m, n, w in itertools.product(
            powers_of_2(10), powers_of_2(10), powers_of_2(3)
        )
        if m * n <= 1 << 13  # Arbitrary choice.
    ]


def generate_config_1d():
    if knobs.runtime.interpret:
        return [triton.Config(kwargs={"BLOCK_SIZE": 4})]
    return [
        triton.Config(kwargs={"BLOCK_SIZE": bsz}, num_warps=w)
        for bsz, w in itertools.product(powers_of_2(10), powers_of_2(3))
    ]


@use_grid(lambda meta: (triton.cdiv((meta['N'] - (meta['k'] + 1)), meta["BLOCK_SIZE"]),))
@derive_launch_arguments(lambda M_real, **_:
                         {'N': M_real.shape[0], })
@triton.autotune(configs=generate_config_1d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_div_column(
        M_real,
        M_imag,
        k, N,
        BLOCK_SIZE: tl.constexpr,
):
    """
    for i in k+1..N-1: A[i,k] /= A[k,k]
    """
    pid = tl.program_id(axis=0)
    rows = k + 1 + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = k

    # pivot
    pivot_real_ptr = M_real + k * N + k
    pivot_imag_ptr = M_imag + k * N + k
    pivot_real = tl.load(pivot_real_ptr)
    pivot_imag = tl.load(pivot_imag_ptr)

    # column to scale
    col_real_ptrs = M_real + rows * N + col
    col_imag_ptrs = M_imag + rows * N + col
    mask = rows < N
    vals_real = tl.load(col_real_ptrs, mask=mask, other=0.0)
    vals_imag = tl.load(col_imag_ptrs, mask=mask, other=0.0)
    vals_real, vals_imag = complex_div(vals_real, vals_imag, pivot_real, pivot_imag)
    tl.store(col_real_ptrs, vals_real, mask=mask)
    tl.store(col_imag_ptrs, vals_imag, mask=mask)


@derive_launch_arguments(lambda M_real, **_:
                         {'N': M_real.shape[0], })
@triton.autotune(configs=generate_config_2d(), key=["N"], cache_results=True)
@triton.jit
def _kernel_lu_trailing_update(
        M_real,
        M_imag,
        k, N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    """
    A[k+1:, k+1:] -= A[k+1:, k] @ A[k, k+1:]   (rank-1 update)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rem = N - (k + 1)
    if rem > 0:
        rows = k + 1 + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        cols = k + 1 + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rm = rows[:, None] < N
        cn = cols[None, :] < N
        mask = rm & cn

        a_real_ptrs = M_real + rows[:, None] * N + cols[None, :]
        a_imag_ptrs = M_imag + rows[:, None] * N + cols[None, :]
        l_real_ptrs = M_real + rows * N + k  # L col k
        l_imag_ptrs = M_imag + rows * N + k  # L col k
        u_real_ptrs = M_real + k * N + cols  # U row k
        u_imag_ptrs = M_imag + k * N + cols  # U row k

        Ablk_real = tl.load(a_real_ptrs, mask=mask, other=0.0)
        Ablk_imag = tl.load(a_imag_ptrs, mask=mask, other=0.0)
        Lcol_real = tl.load(l_real_ptrs, mask=rows < N, other=0.0)[:, None]
        Lcol_imag = tl.load(l_imag_ptrs, mask=rows < N, other=0.0)[:, None]
        Urow_real = tl.load(u_real_ptrs, mask=cols < N, other=0.0)[None, :]
        Urow_imag = tl.load(u_imag_ptrs, mask=cols < N, other=0.0)[None, :]

        temp_real, temp_imag = complex_mul(Lcol_real, Lcol_imag, Urow_real, Urow_imag)
        tl.store(a_real_ptrs, Ablk_real - temp_real, mask=mask)
        tl.store(a_imag_ptrs, Ablk_imag - temp_imag, mask=mask)


@use_grid(lambda meta: (triton.cdiv(meta['NM'], meta['BLOCK_SIZE_M']),))
@derive_launch_arguments(lambda M_real, A_real, **_:
                         {
                             'N': M_real.shape[0],
                             'NM': A_real.shape[-1],
                         })
@triton.autotune(configs=generate_config_2d(), key=["N", "NM"], cache_results=True)
@triton.jit
def _kernel_forward_row(
        M_real, M_imag,  # (NR, NR)
        A_real, A_imag,  # (NR, NM)
        y_real, y_imag,  # (NR, NM)
        N: tl.constexpr, NM: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
):
    """
    Compute: y[i] = b[i] - dot(A[i, :i], y[:i])
    L has unit diagonal, so no division here.
    """
    m = tl.program_id(axis=0)

    for i in range(N):
        acc_real = tl.zeros((BLOCK_SIZE_M,), dtype=M_real.dtype.element_ty)
        acc_imag = tl.zeros((BLOCK_SIZE_M,), dtype=M_real.dtype.element_ty)

        # process in tiles of BLOCK_SIZE
        # num full/partial tiles = ceil(i / BLOCK_SIZE)
        num_tiles = (i + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        for t in range(0, num_tiles):
            tile, mask, rows, _ = get_2d_tile_offsets(m * BLOCK_SIZE_M,
                                                      t * BLOCK_SIZE_N,
                                                      tile_width=BLOCK_SIZE_M,
                                                      tile_height=BLOCK_SIZE_N,
                                                      matrix_width=NM,
                                                      matrix_height=N,
                                                      )  # (BLOCK_SIZE_N, BLOCK_SIZE_M)
            mask &= rows[:, None] < i
            cols = rows
            cols_mask = cols < i
            a_vals_real = tl.load(M_real + i * N + cols, mask=cols_mask, other=0.0)[:, None]
            a_vals_imag = tl.load(M_imag + i * N + cols, mask=cols_mask, other=0.0)[:, None]

            y_vals_real = tl.load(y_real + tile, mask=mask, other=0.0)
            y_vals_imag = tl.load(y_imag + tile, mask=mask, other=0.0)

            mul_real, mul_imag = complex_mul(a_vals_real, a_vals_imag, y_vals_real, y_vals_imag)

            acc_real += tl.sum(mul_real, axis=0)
            acc_imag += tl.sum(mul_imag, axis=0)

        tile, mask, _, _ = get_2d_tile_offsets(x=m * BLOCK_SIZE_M,
                                               y=i,
                                               tile_width=BLOCK_SIZE_M,
                                               tile_height=1,
                                               matrix_width=NM,
                                               matrix_height=N,
                                               )

        bi_real = tl.load(A_real + tile, mask)
        bi_imag = tl.load(A_imag + tile, mask)
        yi_real = bi_real - acc_real
        yi_imag = bi_imag - acc_imag
        tl.store(y_real + tile, yi_real, mask)
        tl.store(y_imag + tile, yi_imag, mask)


@use_grid(lambda meta: (triton.cdiv(meta['NM'], meta['BLOCK_SIZE_M']),))
@derive_launch_arguments(lambda M_real, y_real, **_:
                         {
                             'N': M_real.shape[0],
                             'NM': y_real.shape[-1],
                         })
@triton.autotune(configs=generate_config_2d(), key=["N", "NM"], cache_results=True)
@triton.jit
def _kernel_backward_row(
        M_real, M_imag,  # (NR, NR)
        y_real, y_imag,  # (NR, NM)
        x_real, x_imag,  # (NR, NM)
        N: tl.constexpr,
        NM: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
):
    """
    Compute: x[i] = (y[i] - dot(A[i, i+1:], x[i+1:])) / A[i,i]
    """
    m = tl.program_id(axis=0)
    for i in range(N - 1, -1, -1):
        acc_real = tl.zeros((BLOCK_SIZE_M,), dtype=M_real.dtype.element_ty)
        acc_imag = tl.zeros((BLOCK_SIZE_M,), dtype=M_real.dtype.element_ty)

        # length of the suffix
        len_suf = N - (i + 1)
        num_tiles = (len_suf + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

        for t in range(0, num_tiles):
            tile, mask, rows, _ = get_2d_tile_offsets(m * BLOCK_SIZE_M,
                                                      (i + 1) + t * BLOCK_SIZE_N,
                                                      tile_width=BLOCK_SIZE_M,
                                                      tile_height=BLOCK_SIZE_N,
                                                      matrix_width=NM,
                                                      matrix_height=N,
                                                      )  # (BLOCK_SIZE_N, BLOCK_SIZE_M)

            cols = rows
            cols_mask = cols < N
            a_vals_real = tl.load(M_real + i * N + cols, mask=cols_mask, other=0.0)[:, None]
            a_vals_imag = tl.load(M_imag + i * N + cols, mask=cols_mask, other=0.0)[:, None]
            x_vals_real = tl.load(x_real + tile, mask=mask, other=0.0)
            x_vals_imag = tl.load(x_imag + tile, mask=mask, other=0.0)

            mul_real, mul_imag = complex_mul(a_vals_real, a_vals_imag, x_vals_real, x_vals_imag)

            acc_real += tl.sum(mul_real, axis=0)
            acc_imag += tl.sum(mul_imag, axis=0)

        tile, mask, _, _ = get_2d_tile_offsets(m * BLOCK_SIZE_M,
                                               i,
                                               tile_width=BLOCK_SIZE_M,
                                               tile_height=1,
                                               matrix_width=NM,
                                               matrix_height=N,
                                               )  # (BLOCK_SIZE_N, BLOCK_SIZE_M)

        yi_real = tl.load(y_real + tile, mask)
        yi_imag = tl.load(y_imag + tile, mask)

        aii_real = tl.load(M_real + i * N + i)
        aii_imag = tl.load(M_imag + i * N + i)

        xi_real, xi_imag = complex_div(yi_real - acc_real, yi_imag - acc_imag, aii_real, aii_imag)
        tl.store(x_real + tile, xi_real, mask)
        tl.store(x_imag + tile, xi_imag, mask)


def _linalg_solve(M_real,  # (NR, NR)
                  M_imag,  # (NR, NR)
                  A_real,  # (NR, NM)
                  A_imag,  # (NR, NM)
                  X_real,  # (NR, NM)
                  X_imag,  # (NR, NM)
                  y_real,  # (NR, NM)
                  y_imag,  # (NR, NM)
                  ):
    """
    Solves for every X in: \forall nm: M * X_{nm} = A_{nm}
    """
    N = M_real.shape[0]

    # -------- LU factorization (in-place) --------
    for k in range(N):
        # 1) scale column below pivot
        if k + 1 < N:
            _kernel_lu_div_column(
                M_real, M_imag,
                k,
            )

        # 2) rank-1 update of trailing block
        rem = N - (k + 1)
        if rem > 0:
            grid_upd = lambda meta: (
                triton.cdiv(rem, meta["BLOCK_SIZE_M"]),
                triton.cdiv(rem, meta["BLOCK_SIZE_N"]),
            )

            _kernel_lu_trailing_update[grid_upd](
                M_real, M_imag, k,
            )

    # -------- Forward solve Ly=b (unit lower) --------

    _kernel_forward_row(
        M_real, M_imag,
        A_real, A_imag, y_real, y_imag,
    )

    # -------- Backward solve Ux=y --------

    _kernel_backward_row(
        M_real, M_imag,
        y_real, y_imag,
        X_real, X_imag,
    )


@use_grid(lambda meta: (triton.cdiv(meta['NR'], meta['BLOCK_SIZE_N']),
                        triton.cdiv(meta['NM'], meta['BLOCK_SIZE_M'])))
@derive_launch_arguments(lambda X_real, **_:
                         {
                             'NR': X_real.shape[0],
                             'NM': X_real.shape[1],
                         })
@triton.autotune(configs=generate_config_2d(), key=["NR", "NM"], cache_results=True)
@triton.jit(do_not_specialize=['z_real', 'z_imag'])
def _post_process(
        X_real,  # (NR, NM)
        X_imag,  # (NR, NM)
        P0,  # (NR, NM, 2)
        P1,  # (NR, NM, 2)
        z_real: tl_float,
        z_imag: tl_float,
        NR: tl.constexpr,
        NM: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
):
    n = tl.program_id(axis=0)
    m = tl.program_id(axis=1)

    tile, mask, _, _ = get_2d_tile_offsets(m * BLOCK_SIZE_M, n * BLOCK_SIZE_N,
                                           tile_width=BLOCK_SIZE_M, tile_height=BLOCK_SIZE_N,
                                           matrix_width=NM, matrix_height=NR)
    x_real = tl.load(X_real + tile, mask)
    x_imag = tl.load(X_imag + tile, mask)
    comp_abs = z_real * z_real + z_imag * z_imag
    if comp_abs < 1.0:
        x_real = -x_real
        x_imag = -x_imag

    tile, mask, _, _ = get_2d_tile_offsets(m * BLOCK_SIZE_M * 2,
                                           n * BLOCK_SIZE_N,
                                           tile_width=BLOCK_SIZE_M * 2,
                                           tile_height=BLOCK_SIZE_N,
                                           matrix_width=NM * 2, matrix_height=NR)
    p0 = tl.load(P0 + tile, mask)
    p1 = tl.load(P1 + tile, mask)
    p0 += tl.interleave(x_real, x_imag)
    x_real, x_imag = complex_mul(x_real, x_imag, z_real, z_imag)
    p1 += tl.interleave(x_real, x_imag)
    tl.store(P0 + tile, p0, mask)
    tl.store(P1 + tile, p1, mask)


@use_grid(lambda meta: (triton.cdiv(meta['NR'], meta['BLOCK_SIZE']),
                        triton.cdiv(meta['NR'], meta['BLOCK_SIZE'])))
@derive_launch_arguments(lambda Ham_real, **_:
                         {
                             'NR': Ham_real.shape[-1],
                             'slab_per_bc': Ham_real.shape[0] - 1,
                         })
@triton.autotune(configs=[
    triton.Config(kwargs={"BLOCK_SIZE": bsz}, num_warps=w)
    for bsz, w in itertools.product(powers_of_2(7), powers_of_2(3))
], key=["NR"], cache_results=True)
@triton.jit
def _calculate_tz(
        Tz_real,  # (NR, NR)
        Tz_imag,  # (NR, NR)
        Ham_real,  # (slab_per_bc + 1, NR, NR)
        Ham_imag,  # (slab_per_bc + 1, NR, NR)
        z_real: tl_float,
        z_imag: tl_float,
        NR: tl.constexpr,
        slab_per_bc: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """
        Tz = torch.zeros((NR, NR), dtype=dtype)
        for n in range(slab_per_bc + 1): # Runs 3 times.
            zz = torch.pow(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]

        Tz_real, Tz_imag = Tz.real.contiguous(), Tz.imag.contiguous()
    """
    x = tl.program_id(axis=0)
    y = tl.program_id(axis=1)

    tile, mask, _, _ = get_2d_tile_offsets(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                           tile_width=BLOCK_SIZE,
                                           tile_height=BLOCK_SIZE,
                                           matrix_width=NR,
                                           matrix_height=NR)

    acc_real = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=Tz_real.dtype.element_ty)
    acc_imag = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=Tz_real.dtype.element_ty)
    for n in range(slab_per_bc + 1):
        power = slab_per_bc / 2 - n
        r = tl.sqrt(z_real * z_real + z_imag * z_imag)
        delta = libdevice.atan2(z_imag, z_real)
        r = libdevice.pow(r, power)
        delta *= power
        zz_real = r * tl.cos(delta)
        zz_imag = r * tl.sin(delta)

        ham_real_n = Ham_real + n * NR * NR
        ham_imag_n = Ham_imag + n * NR * NR

        ham_real = tl.load(ham_real_n + tile, mask)
        ham_imag = tl.load(ham_imag_n + tile, mask)
        tmp_real, tmp_imag = complex_mul(zz_real, zz_imag, ham_real, ham_imag)
        acc_real += tmp_real
        acc_imag += tmp_imag

    tl.store(Tz_real + tile, acc_real, mask)
    tl.store(Tz_imag + tile, acc_imag, mask)


def contour_integral(NR,
                     NM,
                     _,
                     Ham,  # (slab_per_bc + 1, NR, NR)[complex128]
                     int_pts: torch.Tensor,  # (num_int_ptsm, )[complex128]
                     Y,  # (NR, NM)[complex128]
                     ):
    dtype = Ham.dtype
    sdtype = torch.float32 if dtype == torch.complex64 else torch.float64
    P0 = torch.zeros_like(Y)
    P0_real = torch.view_as_real(P0)
    P1 = torch.zeros_like(Y)
    P1_real = torch.view_as_real(P1)
    tmp_y_real = torch.empty((NR, NM), dtype=sdtype, device=Y.device)
    tmp_y_imag = torch.empty((NR, NM), dtype=sdtype, device=Y.device)
    X_real = torch.empty_like(Y, dtype=sdtype)
    X_imag = torch.empty_like(Y, dtype=sdtype)

    # TODO: Could consider fusing this into '_kernel_forward_row' if it is too expensive.
    Y_real, Y_imag = Y.real.contiguous(), Y.imag.contiguous()
    # TODO: Could fuse into '_calculate_tz', but likely worse for performance.
    Ham_real, Ham_imag = Ham.real.contiguous(), Ham.imag.contiguous()

    Tz_real = torch.empty((NR, NR), dtype=sdtype, device=Y.device)
    Tz_imag = torch.empty((NR, NR), dtype=sdtype, device=Y.device)
    # Note: 'int_pts' is on the GPU and should be copied to the CPU as one batch for python iteration, otherwise
    #       PyTorch performs needless CUDA synchronization.
    ints = int_pts.tolist()
    for z in ints:
        _calculate_tz(Tz_real, Tz_imag, Ham_real, Ham_imag, float(z.real), float(z.imag))

        X_real.zero_()
        X_imag.zero_()
        _linalg_solve(Tz_real, Tz_imag, Y_real, Y_imag, X_real, X_imag, tmp_y_real, tmp_y_imag)

        # TODO: Consider fusing this into backward row. Would save on all the loads of X within '_post_process', but
        #  not change anything else (in particular peak memory consumption). Profile first! Only guaranteed to improve
        #  performance if memory bound. Could be worse in performance if compute bound.
        _post_process(X_real, X_imag, P0_real, P1_real, float(z.real), float(z.imag))

    return P0, P1
