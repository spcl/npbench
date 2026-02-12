import itertools
import operator
from functools import reduce

import torch
import triton
import triton.language as tl

from npbench.infrastructure.triton_utilities import get_4d_tile_offsets, derive_launch_arguments, use_grid, \
    kernel_mean_and_sumsq, kernel_compute_stddev, get_2d_tile_offsets


def _generate_conv2d_config():
    return [triton.Config(
        kwargs={'BLOCK_SIZE_C1': block_size_c1, 'BLOCK_SIZE_C2': block_size_c2, 'REUSE_INPUT': reuse_input},
        num_warps=warps)
        for block_size_c1, block_size_c2, warps, reuse_input in
        itertools.product([1, 2, 4, 8, 16, 32, 64],
                          [1, 2, 4, 8, 16, 32, 64],
                          [1, 2],
                          [False, True])
        if (block_size_c2 < 512 and warps < 4 if reuse_input else block_size_c1 < 8)]


@use_grid(lambda meta: (meta['H'], meta['W'],
                        meta['N'] * (triton.cdiv(meta['C1'], meta['BLOCK_SIZE_C1']) if meta['REUSE_INPUT']
                        else triton.cdiv(meta['C2'], meta['BLOCK_SIZE_C2']))))
@derive_launch_arguments(lambda input, weights, **_: {
    'N': input.shape[0],
    'H': input.shape[1],
    'W': input.shape[2],
    'C1': input.shape[3],
    'C2': weights.shape[-1],
    'K': weights.shape[0],
    'K_NEXT_2': triton.next_power_of_2(weights.shape[0])
})
@triton.autotune(configs=_generate_conv2d_config(),
                 key=['N', 'H', 'W', 'K', 'C1', 'C2'],
                 cache_results=True
                 )
@triton.jit()
def _conv2d(input,  # (N, H, W, C1)
            weights,  # (K, K, C1, C2)
            output,  # (N, H - K + 1, W - K + 1, C2),
            N: tl.constexpr,
            H: tl.constexpr,
            W: tl.constexpr,
            K: tl.constexpr,
            C1: tl.constexpr,
            C2: tl.constexpr,
            K_NEXT_2: tl.constexpr,
            BLOCK_SIZE_C1: tl.constexpr = 1,
            BLOCK_SIZE_C2: tl.constexpr = 16,
            REUSE_INPUT: tl.constexpr = False,
            ):
    """
    for i in range(H_out): # 56
        for j in range(W_out): # 56
            for n in range(N): # 8
                for c1 in range(C_in): # 256
                    for c2 in range(C_out): # 256
                        output[n, i, j, c2] +=
                            input[n, i:i + K, j:j + K, c1] *
                            weights[:, :, c1, c2]
    """
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    extra = tl.program_id(axis=2)
    n = extra % N

    H_out = H - K + 1
    W_out = H - K + 1

    # Depending on the shape of the input, a specific order of the 'c1' and 'c2' loops might be faster than the other.
    # We perform auto-tuning that accounts for this.
    # Depending on which version is taken we either are parallelizing over 'c1' or 'c2'. In the former case a parallel
    # Reduction is performed that requires the 'output' tensor to be zero initialized.
    if REUSE_INPUT:
        c1 = extra // N

        input_tile, input_mask = get_4d_tile_offsets(
            n, i, j, c1 * BLOCK_SIZE_C1,
            tile_dims=(1, K_NEXT_2, K_NEXT_2, BLOCK_SIZE_C1),
            matrix_dims=(N, H, W, C1),
        )
        conv_matrix = tl.load(
            input + input_tile,
            input_mask,
            other=0.0,
        ).reshape(K_NEXT_2 * K_NEXT_2 * BLOCK_SIZE_C1, 1)

        for c2 in range(tl.cdiv(C2, BLOCK_SIZE_C2)):
            tile, mask = get_4d_tile_offsets(
                0, 0, c1 * BLOCK_SIZE_C1, c2 * BLOCK_SIZE_C2,
                tile_dims=(K_NEXT_2, K_NEXT_2, BLOCK_SIZE_C1, BLOCK_SIZE_C2),
                matrix_dims=(K, K, C1, C2),
            )
            weight_tile = tl.load(weights + tile, mask, other=0.0).reshape(K_NEXT_2 * K_NEXT_2 * BLOCK_SIZE_C1,
                                                                           BLOCK_SIZE_C2)
            sum = tl.sum(conv_matrix * weight_tile, axis=0)[None, None, None, :]

            output_tile, output_mask = get_4d_tile_offsets(
                n, i, j, c2 * BLOCK_SIZE_C2,
                tile_dims=(1, 1, 1, BLOCK_SIZE_C2),
                matrix_dims=(N, H_out, W_out, C2),
            )
            tl.atomic_add(output + output_tile, sum, output_mask)
    else:
        c2 = extra // N

        sum = tl.zeros(shape=(1, 1, 1, BLOCK_SIZE_C2), dtype=input.dtype.element_ty)
        for c1 in tl.range(tl.cdiv(C1, BLOCK_SIZE_C1)):
            input_tile, input_mask = get_4d_tile_offsets(
                n, i, j, c1 * BLOCK_SIZE_C1,
                tile_dims=(1, K_NEXT_2, K_NEXT_2, BLOCK_SIZE_C1),
                matrix_dims=(N, H, W, C1),
            )
            conv_matrix = tl.load(
                input + input_tile,
                input_mask,
                other=0.0,
            ).reshape(K_NEXT_2 * K_NEXT_2 * BLOCK_SIZE_C1, 1)

            tile, mask = get_4d_tile_offsets(
                0, 0, c1 * BLOCK_SIZE_C1, c2 * BLOCK_SIZE_C2,
                tile_dims=(K_NEXT_2, K_NEXT_2, BLOCK_SIZE_C1, BLOCK_SIZE_C2),
                matrix_dims=(K, K, C1, C2),
            )
            weight_tile = tl.load(weights + tile, mask, other=0.0).reshape(K_NEXT_2 * K_NEXT_2 * BLOCK_SIZE_C1,
                                                                           BLOCK_SIZE_C2)
            sum += tl.sum(conv_matrix * weight_tile, axis=0)[None, None, None, :]

        output_tile, output_mask = get_4d_tile_offsets(
            n, i, j, c2 * BLOCK_SIZE_C2,
            tile_dims=(1, 1, 1, BLOCK_SIZE_C2),
            matrix_dims=(N, H_out, W_out, C2),
        )
        tl.store(output + output_tile, sum, output_mask)


@use_grid(lambda meta: (
        triton.cdiv(meta['N'], meta['BLOCK_SIZE_N']),
        triton.cdiv(meta['M'], meta['BLOCK_SIZE_M']),
))
@derive_launch_arguments(lambda x, **_: {
    'N': reduce(operator.mul, x.shape[1:], 1),
    'M': x.shape[0],
})
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE_N': n, 'BLOCK_SIZE_M': m}, num_warps=w)
    for n, m, w in
    itertools.product([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                      [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                      [1, 2, 4, 8])
    if n * m < (1 << 16)  # Arbitrary limit to not be too slow.
],
    key=['N', 'M'],
    cache_results=True
)
@triton.jit()
def _batchnorm2d_normalize(x,  # (M, N)
                           mean,  # (N,)
                           stddev,  # (N,)
                           eps,
                           N: tl.constexpr,
                           M: tl.constexpr,
                           BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr,
                           post_process: tl.constexpr,
                           extra_arg=0.0,
                           ):
    n = tl.program_id(axis=0)
    m = tl.program_id(axis=1)
    tile, mask, rows, columns = get_2d_tile_offsets(n * BLOCK_SIZE_N,
                                                    m * BLOCK_SIZE_M,
                                                    tile_width=BLOCK_SIZE_N,
                                                    tile_height=BLOCK_SIZE_M,
                                                    matrix_width=N,
                                                    matrix_height=M)
    x_tile = tl.load(x + tile, mask)
    mean_tile = tl.load(mean + columns, columns < N)
    std_tile = tl.load(stddev + columns, columns < N)
    result = post_process((x_tile - mean_tile) * tl.rsqrt(std_tile + eps), tile, mask, extra_arg)
    tl.store(x + tile, result, mask)


def _padded_batchnorm2d_relu(x, eps=1e-5):
    """
    Fused implementation of batchnorm2d with relu activation with a padding preprocessing step.
    """

    padded = torch.zeros((x.shape[0], x.shape[1] + 2, x.shape[2] + 2,
                          x.shape[3]), dtype=x.dtype, device=x.device)
    # TODO: Maybe this can somehow be fused into 'batchnorm2d_relu'?
    padded[:, 1:-1, 1:-1, :] = x
    return _batchnorm2d_relu(padded, eps)


# Batch normalization operator, as used in ResNet
def _batchnorm2d_relu_input(x,  # (N, H, W, C)
                            input,  # (N, H, W, C)
                            eps=1e-5):
    """
    Fused implementation of batchnorm2d with 'relu(result + input)' activation.
    """

    N, H, W, C = x.shape
    mean = torch.zeros((1, H, W, C), dtype=x.dtype)
    stddev = torch.zeros((1, H, W, C), dtype=x.dtype)

    # (N, H, W, C) -> (1, H, W, C)
    kernel_mean_and_sumsq(x, mean, stddev)
    # (N, H, W, C) -> (1, H, W, C)
    kernel_compute_stddev(mean, stddev)

    @triton.jit()
    def post_process(x, tile, mask, input):
        input_tile = tl.load(input + tile, mask)
        return tl.maximum(x + input_tile, 0.0)

    # (N, H, W, C) -> (1, H, W, C) -> (1, H, W, C) -> () -> (N, H, W, C)
    _batchnorm2d_normalize(x, mean, stddev, eps, post_process=post_process, extra_arg=input)
    return x


def _batchnorm2d_relu(x,  # (N, H, W, C)
                      eps=1e-5):
    """
    Fused implementation of batchnorm2d with relu activation.
    """

    N, H, W, C = x.shape
    mean = torch.zeros((1, H, W, C), dtype=x.dtype)
    stddev = torch.zeros((1, H, W, C), dtype=x.dtype)

    # (N, H, W, C) -> (1, H, W, C)
    kernel_mean_and_sumsq(x, mean, stddev)
    # (N, H, W, C) -> (1, H, W, C)
    kernel_compute_stddev(mean, stddev)

    @triton.jit()
    def post_process(x, _0, _1, _2): return tl.maximum(x, 0.0)

    # (N, H, W, C) -> (1, H, W, C) -> (1, H, W, C) -> () -> (N, H, W, C)
    _batchnorm2d_normalize(x, mean, stddev, eps, post_process=post_process)
    return x


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3):
    N, H, W, C1 = input.shape
    C2 = conv1.shape[-1]

    x_new = torch.zeros((N, H, W, C2), dtype=input.dtype,
                        device=input.device)
    # (N, H, W, C1) -> (1, 1, C1, C2) -> (N, H, W, C2)
    _conv2d(input, conv1, x_new)

    # (N, H + 2, W + 2, C2) -> (N, H + 2, W + 2, C2)
    x2 = _padded_batchnorm2d_relu(x_new)

    # Required by convolution implementation.
    x_new[:] = 0
    # (N, H + 2, W + 2, C2) -> (3, 3, C2, C2) -> (N, H, W, C2)
    _conv2d(x2, conv2, x_new)

    x = _batchnorm2d_relu(x_new)

    x_new = torch.zeros_like(input)
    # (N, H, W, C2) -> (1, 1, C2, C1) -> (N, H, W, C1)
    _conv2d(x, conv3, x_new)
    # (N, H, W, C1) -> (N, H, W, C1) -> (N, H, W, C1)
    return _batchnorm2d_relu_input(x_new, input)
