import itertools
import torch
import triton
import triton.language as tl


def get_conv2d_configs():
    return [
        triton.Config({"BLOCK_C_IN": bc}, num_warps=w)
        for bc, w in itertools.product(
            [32, 64, 128, 256],  # BLOCK_C_IN options
            [2, 4, 8]            # num_warps options
        )
    ]


@triton.autotune(
    configs=get_conv2d_configs(),
    key=["C_in", "C_out", "K"],
    cache_results=True
)
@triton.jit
def _kernel_conv2d(
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
    tl.store(output_ptr + output_idx, acc)


def conv2d_bias(input, weights, bias):
    N, H, W, C_in = input.shape
    K = weights.shape[0]
    C_out = weights.shape[3]
    H_out = H - K + 1
    W_out = W - K + 1

    output = torch.empty((N, H_out, W_out, C_out), device=input.device, dtype=input.dtype)

    grid = (N * H_out * W_out, C_out, 1)
    _kernel_conv2d[grid](
        input, weights, output, bias,
        N, H, W, C_in,
        K,
        C_out,
        H_out, W_out,
    )

    return output