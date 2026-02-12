import itertools
import torch
import triton
import triton.language as tl

from npbench.infrastructure.triton_utilities import matmul


def get_conv2d_configs():
    return [
        triton.Config({"BLOCK_C_IN": bc}, num_warps=w)
        for bc, w in itertools.product(
            [16, 32, 64, 128],
            [2, 4, 8]
        )
    ]


@triton.autotune(
    configs=get_conv2d_configs(),
    key=["C_in", "C_out", "K"],
    cache_results=True
)
@triton.jit
def _kernel_conv2d_bias_relu(
        input_ptr,
        weights_ptr,
        output_ptr,
        bias_ptr,
        N, H, W, C_in,
        K,
        C_out,
        H_out, W_out,
        BLOCK_C_IN: tl.constexpr,
):
    spatial_idx = tl.program_id(0)
    c_out = tl.program_id(1)

    n = spatial_idx // (H_out * W_out)
    remainder = spatial_idx % (H_out * W_out)
    h_out = remainder // W_out
    w_out = remainder % W_out

    acc = tl.load(bias_ptr + c_out)

    for kh in range(K):
        for kw in range(K):
            for c_in_block_start in range(0, C_in, BLOCK_C_IN):
                c_in_offsets = c_in_block_start + tl.arange(0, BLOCK_C_IN)
                c_in_mask = c_in_offsets < C_in

                input_base = n * H * W * C_in + (h_out + kh) * W * C_in + (w_out + kw) * C_in
                input_indices = input_base + c_in_offsets
                input_vals = tl.load(input_ptr + input_indices, mask=c_in_mask, other=0.0)

                weight_base = kh * K * C_in * C_out + kw * C_in * C_out + c_out
                weight_indices = weight_base + c_in_offsets * C_out
                weight_vals = tl.load(weights_ptr + weight_indices, mask=c_in_mask, other=0.0)

                acc += tl.sum(input_vals * weight_vals)

    output_idx = n * H_out * W_out * C_out + h_out * W_out * C_out + w_out * C_out + c_out
    tl.store(output_ptr + output_idx, tl.maximum(acc, 0.0))


def conv2d_bias_relu(input, weights, bias):
    N, H, W, C_in = input.shape
    K = weights.shape[0]
    C_out = weights.shape[3]
    H_out = H - K + 1
    W_out = W - K + 1

    output = torch.empty((N, H_out, W_out, C_out), device=input.device, dtype=input.dtype)

    grid = (N * H_out * W_out, C_out, 1)
    _kernel_conv2d_bias_relu[grid](
        input, weights, output, bias,
        N, H, W, C_in,
        K,
        C_out,
        H_out, W_out,
    )

    return output


def get_maxpool_configs():
    return [
        triton.Config({"BLOCK_C": bc}, num_warps=w)
        for bc, w in itertools.product(
            [4, 8, 16, 32],
            [1, 2, 4]
        )
    ]


@triton.autotune(
    configs=get_maxpool_configs(),
    key=["C"],
    cache_results=True
)
@triton.jit
def _kernel_maxpool2d(
        input_ptr,
        output_ptr,
        N, H, W, C,
        H_out, W_out,
        BLOCK_C: tl.constexpr,
):
    spatial_idx = tl.program_id(0)
    c_block_start = tl.program_id(1) * BLOCK_C

    n = spatial_idx // (H_out * W_out)
    remainder = spatial_idx % (H_out * W_out)
    h_out = remainder // W_out
    w_out = remainder % W_out

    c_offsets = c_block_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    h_in = h_out * 2
    w_in = w_out * 2

    max_val = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)

    for i in range(2):
        for j in range(2):
            input_base = n * H * W * C + (h_in + i) * W * C + (w_in + j) * C
            input_indices = input_base + c_offsets
            input_vals = tl.load(input_ptr + input_indices, mask=c_mask, other=-float('inf'))
            max_val = tl.maximum(max_val, input_vals)

    output_base = n * H_out * W_out * C + h_out * W_out * C + w_out * C
    output_indices = output_base + c_offsets
    tl.store(output_ptr + output_indices, max_val, mask=c_mask)


def maxpool2d(x):
    N, H, W, C = x.shape
    H_out = H // 2
    W_out = W // 2

    output = torch.empty((N, H_out, W_out, C), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N * H_out * W_out, triton.cdiv(C, meta["BLOCK_C"]), 1)
    _kernel_maxpool2d[grid](
        x, output,
        N, H, W, C,
        H_out, W_out,
    )

    return output


def get_fc_configs():
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for bs, w in itertools.product(
            [8, 16, 32, 64, 128],
            [1, 2, 4, 8]
        )
    ]


@triton.autotune(configs=get_fc_configs(), key=["N"], cache_results=True)
@triton.jit
def _kernel_bias_relu(
        A_ptr,
        B_ptr,
        N: tl.int32,
        stride_am: tl.int32,
        stride_an: tl.int32,
        BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    offsets_a = pid_m * stride_am + cols * stride_an
    offsets_b = cols

    a = tl.load(A_ptr + offsets_a, mask=mask)
    b = tl.load(B_ptr + offsets_b, mask=mask)

    out = tl.maximum(a + b, 0.0)

    tl.store(A_ptr + offsets_a, out, mask=mask)


def fc_bias_relu(A, B):
    M, N = A.shape

    grid = lambda meta: (
        M,
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )

    _kernel_bias_relu[grid](A, B, N, A.stride(0), A.stride(1))


@triton.autotune(configs=get_fc_configs(), key=["N"], cache_results=True)
@triton.jit
def _kernel_bias(
        A_ptr,
        B_ptr,
        N: tl.int32,
        stride_am: tl.int32,
        stride_an: tl.int32,
        BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    offsets_a = pid_m * stride_am + cols * stride_an
    offsets_b = cols

    a = tl.load(A_ptr + offsets_a, mask=mask)
    b = tl.load(B_ptr + offsets_b, mask=mask)

    tl.store(A_ptr + offsets_a, a + b, mask=mask)


def fc_bias(A, B):
    M, N = A.shape

    grid = lambda meta: (
        M,
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )

    _kernel_bias[grid](A, B, N, A.stride(0), A.stride(1))


def lenet5(input, conv1, conv1bias, conv2, conv2bias, fc1w, fc1b, fc2w, fc2b,
           fc3w, fc3b, N, C_before_fc1):
    x = conv2d_bias_relu(input, conv1, conv1bias)
    x = maxpool2d(x)
    x = conv2d_bias_relu(x, conv2, conv2bias)
    x = maxpool2d(x)

    x = x.reshape(N, C_before_fc1)

    x = matmul(x, fc1w)
    fc_bias_relu(x, fc1b)

    y = matmul(x, fc2w)
    fc_bias_relu(y, fc2b)

    z = matmul(y, fc3w)
    fc_bias(z, fc3b)

    return z
