import torch
import triton
import triton.language as tl
import itertools

def get_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_X": bx, "BLOCK_SIZE_Y": by}, num_warps=w
        ) for bx, by, w in itertools.product([4, 8, 16, 32], [4, 8, 16], [1, 2, 4, 8])
    ]

@triton.autotune(configs=get_configs(), key=["xn", "yn", "maxiter"], cache_results=True)
@triton.jit
def _kernel_mandelbrot(
      N_ptr, 
      Z_real_ptr,
      Z_imag_ptr,
      xmin: tl.float64, xmax: tl.float64, ymin: tl.float64, ymax: tl.float64,
      xn, yn,
      maxiter,
      horizon,
      BLOCK_SIZE_X: tl.constexpr,
      BLOCK_SIZE_Y: tl.constexpr,
  ):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    x_idx = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_idx = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    x_mask = x_idx < xn
    y_mask = y_idx < yn
    mask_2d = x_mask[None, :] & y_mask[:, None]

    x_coords = xmin + x_idx * (xmax - xmin) / (xn - 1.0)
    y_coords = ymin + y_idx * (ymax - ymin) / (yn - 1.0)

    C_real = x_coords[None, :]
    C_imag = y_coords[:, None]
    
    Z_real_current = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)
    Z_imag_current = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)
    
    N_out = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.int64)
    Z_real = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)
    Z_imag = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float64)

    active_mask_int = tl.full((BLOCK_SIZE_Y, BLOCK_SIZE_X), value=1, dtype=tl.int32)
    horizon_sq = horizon * horizon

    for n in range(maxiter):
        Z_real_new = Z_real_current * Z_real_current - Z_imag_current * Z_imag_current + C_real
        Z_imag_new = 2.0 * Z_real_current * Z_imag_current + C_imag

        Z_abs_sq = Z_real_new * Z_real_new + Z_imag_new * Z_imag_new
        failed_mask = (Z_abs_sq > horizon_sq)
        
        just_failed_mask = failed_mask & (active_mask_int == 1)
        
        N_out = tl.where(just_failed_mask, n + 1, N_out)
        
        Z_real = tl.where(just_failed_mask, Z_real_new, Z_real)
        Z_imag = tl.where(just_failed_mask, Z_imag_new, Z_imag)

        active_mask_int = tl.where(just_failed_mask, 0, active_mask_int)
        
        active_mask_bool = active_mask_int == 1
        Z_real_current = tl.where(active_mask_bool, Z_real_new, Z_real_current)
        Z_imag_current = tl.where(active_mask_bool, Z_imag_new, Z_imag_current)

    offsets = y_idx[:, None] * xn + x_idx[None, :]
    tl.store(N_ptr + offsets, N_out, mask=mask_2d)
    tl.store(Z_real_ptr + offsets, Z_real, mask=mask_2d)
    tl.store(Z_imag_ptr + offsets, Z_imag, mask=mask_2d)


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Allocate output tensors
    # X = torch.Tensor(np.linspace(xmin, xmax, xn, dtype=np.float64))
    # Y = torch.Tensor(np.linspace(ymin, ymax, yn, dtype=np.float64))
    # no need for the following as it can be computed inside the kernel: C = torch.Tensor(X + Y[:, None] * 1j)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = torch.zeros((yn, xn), dtype=torch.int64, device=device)
    Z_real = torch.zeros((yn, xn), dtype=torch.float64, device=device)
    Z_imag = torch.zeros((yn, xn), dtype=torch.float64, device=device)

    grid = lambda meta: (
        triton.cdiv(xn, meta['BLOCK_SIZE_X']),
        triton.cdiv(yn, meta['BLOCK_SIZE_Y'])
    )

    _kernel_mandelbrot[grid](
        N,
        Z_real,
        Z_imag,
        xmin,
        xmax,
        ymin,
        ymax,
        xn,
        yn,
        maxiter,
        horizon,
    )
    Z = torch.complex(Z_real, Z_imag)
    return Z, N