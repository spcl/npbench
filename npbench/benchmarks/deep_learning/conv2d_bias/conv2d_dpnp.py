import dpnp as np

# Optimized Deep Learning Convolutional Operator (stride = 1)
def conv2d(input, weights):
    K = weights.shape[0]  # Kernel size (assuming square kernel)
    N, H, W, C_in = input.shape  # Input dimensions
    C_out = weights.shape[3]  # Output channels
    H_out = H - K + 1
    W_out = W - K + 1
    output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

    # Perform convolution manually by iterating over the kernel dimensions
    for i in range(K):
        for j in range(K):
            output += np.tensordot(
                input[:, i:H_out + i, j:W_out + j, :],
                weights[i, j, :, :],
                axes=([3], [0])
            )

    return output

def conv2d_bias(input, weights, bias):
    return conv2d(input, weights) + bias

# Optimized Initialization Function for DPNP
def initialize(C_in, C_out, H, K, N, W):
    from dpnp.random import default_rng
    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out,), dtype=np.float32)
    return input, weights, bias

