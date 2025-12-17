import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config(kwargs={"BLOCK_SIZE_X": bx, "BLOCK_SIZE_Y": by}, num_warps=nw)
        for bx in [8, 16, 32]
        for by in [8, 16, 32]
        for nw in [2, 4, 8, 16]
    ]


@triton.autotune(configs=get_autotune_config(), key=["H", "W"], cache_results=True, )
@triton.jit
def build_b_kernel(
    u_ptr,
    v_ptr,
    b_ptr,
    rho,
    dt,
    dx,
    dy,
    H,
    W,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Mask to stay within bounds.
    # Calculation is for interior points (1:-1), so we exclude 0 and H-1.
    mask_x = pid_x < W
    mask_y = (pid_y > 0) & (pid_y < H - 1)
    mask = mask_y[:, None] & mask_x[None, :]

    # Offsets
    center_ptr = pid_y[:, None] * W + pid_x[None, :]

    # Neighbors with Periodic X handling
    left_x = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y = pid_y[:, None] + 1
    down_y = pid_y[:, None] - 1

    # Load U
    u_r = tl.load(u_ptr + (pid_y[:, None] * W + right_x), mask=mask)
    u_l = tl.load(u_ptr + (pid_y[:, None] * W + left_x), mask=mask)
    u_u = tl.load(u_ptr + (up_y * W + pid_x[None, :]), mask=mask)
    u_d = tl.load(u_ptr + (down_y * W + pid_x[None, :]), mask=mask)

    # Load V
    v_r = tl.load(v_ptr + (pid_y[:, None] * W + right_x), mask=mask)
    v_l = tl.load(v_ptr + (pid_y[:, None] * W + left_x), mask=mask)
    v_u = tl.load(v_ptr + (up_y * W + pid_x[None, :]), mask=mask)
    v_d = tl.load(v_ptr + (down_y * W + pid_x[None, :]), mask=mask)

    # Central Differences
    dudx = (u_r - u_l) / (2.0 * dx)
    dvdy = (v_u - v_d) / (2.0 * dy)
    dudy = (u_u - u_d) / (2.0 * dy)
    dvdx = (v_r - v_l) / (2.0 * dx)

    # Source term calculation
    term1 = (1.0 / dt) * (dudx + dvdy)
    term2 = dudx * dudx
    term3 = 2.0 * (dudy * dvdx)
    term4 = dvdy * dvdy

    result = rho * (term1 - term2 - term3 - term4)
    tl.store(b_ptr + center_ptr, result, mask=mask)


@triton.autotune(configs=get_autotune_config(), key=["H", "W"], cache_results=True, )
@triton.jit
def pressure_poisson_kernel(
    p_new_ptr,
    p_old_ptr,
    b_ptr,
    dx,
    dy,
    H,
    W,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    mask_x = pid_x < W
    mask_y = pid_y < H
    mask = mask_y[:, None] & mask_x[None, :]

    # Identify Regions
    is_wall_bottom = pid_y[:, None] == 0
    is_wall_top = pid_y[:, None] == H - 1
    is_interior = (~is_wall_bottom) & (~is_wall_top)

    # --- Interior Logic ---
    left_x = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y = pid_y[:, None] + 1
    down_y = pid_y[:, None] - 1

    # Load neighbors (from p_old)
    pn_r = tl.load(p_old_ptr + (pid_y[:, None] * W + right_x), mask=mask & is_interior)
    pn_l = tl.load(p_old_ptr + (pid_y[:, None] * W + left_x), mask=mask & is_interior)
    pn_u = tl.load(p_old_ptr + (up_y * W + pid_x[None, :]), mask=mask & is_interior)
    pn_d = tl.load(p_old_ptr + (down_y * W + pid_x[None, :]), mask=mask & is_interior)

    b_val = tl.load(
        b_ptr + (pid_y[:, None] * W + pid_x[None, :]), mask=mask & is_interior
    )

    # Poisson Equation
    dx2 = dx * dx
    dy2 = dy * dy
    p_computed = ((pn_r + pn_l) * dy2 + (pn_u + pn_d) * dx2) / (2 * (dx2 + dy2)) - (
        dx2 * dy2
    ) / (2 * (dx2 + dy2)) * b_val

    # --- Wall Logic ---
    # p[0, :] = p[1, :] (dp/dy = 0)
    # p[-1, :] = p[-2, :]
    val_at_row1 = tl.load(p_old_ptr + (1 * W + pid_x[None, :]), mask=mask_x[None, :])
    val_at_rowHm2 = tl.load(
        p_old_ptr + ((H - 2) * W + pid_x[None, :]), mask=mask_x[None, :]
    )

    # Combine results
    result = tl.where(is_interior, p_computed, 0.0)
    result = tl.where(is_wall_bottom, val_at_row1, result)
    result = tl.where(is_wall_top, val_at_rowHm2, result)

    tl.store(p_new_ptr + (pid_y[:, None] * W + pid_x[None, :]), result, mask=mask)


@triton.autotune(configs=get_autotune_config(), key=["H", "W"], cache_results=True, )
@triton.jit
def update_uv_kernel(
    u_new_ptr,
    v_new_ptr,
    u_old_ptr,
    v_old_ptr,
    p_ptr,
    rho,
    nu,
    dt,
    dx,
    dy,
    F,
    H,
    W,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    pid_y = tl.program_id(1) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Interior only (1:-1), Walls (0 and H-1) remain 0
    mask_x = pid_x < W
    mask_y = (pid_y > 0) & (pid_y < H - 1)
    mask = mask_y[:, None] & mask_x[None, :]

    center_ptr = pid_y[:, None] * W + pid_x[None, :]

    # Indices
    left_x = (pid_x[None, :] - 1 + W) % W
    right_x = (pid_x[None, :] + 1) % W
    up_y = pid_y[:, None] + 1
    down_y = pid_y[:, None] - 1

    # Load Current
    un_c = tl.load(u_old_ptr + center_ptr, mask=mask)
    vn_c = tl.load(v_old_ptr + center_ptr, mask=mask)

    # Load Neighbors
    un_r = tl.load(u_old_ptr + (pid_y[:, None] * W + right_x), mask=mask)
    un_l = tl.load(u_old_ptr + (pid_y[:, None] * W + left_x), mask=mask)
    un_u = tl.load(u_old_ptr + (up_y * W + pid_x[None, :]), mask=mask)
    un_d = tl.load(u_old_ptr + (down_y * W + pid_x[None, :]), mask=mask)

    vn_r = tl.load(v_old_ptr + (pid_y[:, None] * W + right_x), mask=mask)
    vn_l = tl.load(v_old_ptr + (pid_y[:, None] * W + left_x), mask=mask)
    vn_u = tl.load(v_old_ptr + (up_y * W + pid_x[None, :]), mask=mask)
    vn_d = tl.load(v_old_ptr + (down_y * W + pid_x[None, :]), mask=mask)

    p_r = tl.load(p_ptr + (pid_y[:, None] * W + right_x), mask=mask)
    p_l = tl.load(p_ptr + (pid_y[:, None] * W + left_x), mask=mask)
    p_u = tl.load(p_ptr + (up_y * W + pid_x[None, :]), mask=mask)
    p_d = tl.load(p_ptr + (down_y * W + pid_x[None, :]), mask=mask)

    # Updates
    u_adv = un_c * dt / dx * (un_c - un_l) + vn_c * dt / dy * (un_c - un_d)
    u_press = dt / (2 * rho * dx) * (p_r - p_l)

    # Replaced dx**2 with dx*dx and dy**2 with dy*dy
    dx2 = dx * dx
    dy2 = dy * dy

    u_diff = nu * (
        dt / dx2 * (un_r - 2 * un_c + un_l) + dt / dy2 * (un_u - 2 * un_c + un_d)
    )

    v_adv = un_c * dt / dx * (vn_c - vn_l) + vn_c * dt / dy * (vn_c - vn_d)
    v_press = dt / (2 * rho * dy) * (p_u - p_d)
    v_diff = nu * (
        dt / dx2 * (vn_r - 2 * vn_c + vn_l) + dt / dy2 * (vn_u - 2 * vn_c + vn_d)
    )

    tl.store(
        u_new_ptr + center_ptr, un_c - u_adv - u_press + u_diff + F * dt, mask=mask
    )
    tl.store(v_new_ptr + center_ptr, vn_c - v_adv - v_press + v_diff, mask=mask)


def channel_flow(nit, u, v, dt, dx, dy, p, rho, nu, F):
    H, W = u.shape

    b_dev = torch.empty_like(u)
    u_buff = torch.empty_like(u)
    v_buff = torch.empty_like(v)
    p_buff = torch.empty_like(p)

    u_curr, u_next = u, u_buff
    v_curr, v_next = v, v_buff
    p_curr, p_next = p, p_buff

    grid = lambda meta: (
        triton.cdiv(W, meta["BLOCK_SIZE_X"]),
        triton.cdiv(H, meta["BLOCK_SIZE_Y"]),
    )

    udiff = 1.0
    stepcount = 0

    sum_u_curr = torch.sum(u_curr)

    while udiff > 0.001:
        build_b_kernel[grid](u_curr, v_curr, b_dev, float(rho), float(dt), float(dx), float(dy), H, W)

        p_in, p_out = p_curr, p_next
        for _ in range(nit):
            pressure_poisson_kernel[grid](p_out, p_in, b_dev, float(dx), float(dy), H, W)
            p_in, p_out = p_out, p_in  # Swap

        p_curr, p_next = p_in, p_out

        update_uv_kernel[grid](
            u_next, v_next, u_curr, v_curr, p_curr, float(rho), float(nu), float(dt), float(dx), float(dy), float(F), H, W
        )

        sum_u_next = torch.sum(u_next)
        udiff = (sum_u_next - sum_u_curr) / sum_u_next
        sum_u_curr = sum_u_next
        u_curr, u_next = u_next, u_curr
        v_curr, v_next = v_next, v_curr

        stepcount += 1

    if u_curr is not u:
        u.copy_(u_curr)
    if v_curr is not v:
        v.copy_(v_curr)
    if p_curr is not p:
        p.copy_(p_curr)

    return stepcount
