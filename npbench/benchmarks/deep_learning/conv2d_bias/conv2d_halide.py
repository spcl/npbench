import numpy as np
import halide as hl

def conv2d_bias_params():
    input = hl.ImageParam(hl.Float(32), 4, "input")
    weights = hl.ImageParam(hl.Float(32), 4, "weights")
    bias = hl.ImageParam(hl.Float(32), 1, "bias")

    return input, weights, bias

# Deep learning convolutional operator (stride = 1)
def conv2d_bias(input, weights, bias):
    N = 8
    CI = 3
    CO = 16
    W = 256
    H = 256
    K = 20
    border = K - 1

    x = hl.Var("x")
    y = hl.Var("y")
    c = hl.Var("c")
    n = hl.Var("n")

    r = hl.RDom([(0, CI), (0, K), (0, K)])

    output = hl.Func("output")
    output[c, x, y, n] = bias[c]
    output[c, x, y, n] += weights[r.x, r.y, r.z, c] * input[r.x, x + r.y, y + r.z, n]

    # Bounds

    input.dim(0).set_bounds(0, CI).set_stride(1)
    input.dim(1).set_bounds(0, W).set_stride(CI)
    input.dim(2).set_bounds(0, H).set_stride(CI * W)
    input.dim(3).set_bounds(0, N).set_stride(CI * W * H)

    weights.dim(0).set_bounds(0, CI).set_stride(1)
    weights.dim(1).set_bounds(0, K).set_stride(CI)
    weights.dim(2).set_bounds(0, K).set_stride(CI * K)
    weights.dim(3).set_bounds(0, CO).set_stride(CI * K * K)

    bias.dim(0).set_bounds(0, CO).set_stride(1)

    # Estimates

    input.dim(0).set_estimate(0, CI)
    input.dim(1).set_estimate(0, W) 
    input.dim(2).set_estimate(0, H)
    input.dim(3).set_estimate(0, N)

    output.set_estimate(c, 0, CO)
    output.set_estimate(x, 0, W - border)
    output.set_estimate(y, 0, H - border)
    output.set_estimate(n, 0, N)

    weights.dim(0).set_estimate(0, CI)
    weights.dim(1).set_estimate(0, K)
    weights.dim(2).set_estimate(0, K)
    weights.dim(3).set_estimate(0, CO)

    bias.dim(0).set_estimate(0, CO)

    return output
