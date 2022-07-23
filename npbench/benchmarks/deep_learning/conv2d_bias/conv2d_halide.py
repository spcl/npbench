import numpy as np
import halide as hl

input_buffers = {
    "input": hl.ImageParam(hl.Float(32), 4, "input"),
    "weights": hl.ImageParam(hl.Float(32), 4, "weights"),
    "bias": hl.ImageParam(hl.Float(32), 1, "bias")
}

x = hl.Var("x")
y = hl.Var("y")
c = hl.Var("c")
n = hl.Var("n")

def set_estimates(input, weights, bias, output, N, C_in, C_out, W, H, K):
    input.dim(0).set_estimate(0, C_in)
    input.dim(1).set_estimate(0, W) 
    input.dim(2).set_estimate(0, H)
    input.dim(3).set_estimate(0, N)

    output.set_estimate(c, 0, C_out)
    output.set_estimate(x, 0, W - (K - 1))
    output.set_estimate(y, 0, H - (K - 1))
    output.set_estimate(n, 0, N)

    weights.dim(0).set_estimate(0, C_in)
    weights.dim(1).set_estimate(0, K)
    weights.dim(2).set_estimate(0, K)
    weights.dim(3).set_estimate(0, C_out)

    bias.dim(0).set_estimate(0, C_out)

# Deep learning convolutional operator (stride = 1)
def conv2d_bias(input, weights, bias):

    r = hl.RDom([(0, weights.width()), (0, weights.height()), (0, weights.channels())])
    output = hl.Func("output")
    output[c, x, y, n] = bias[c]
    output[c, x, y, n] += weights[r.x, r.y, r.z, c] * input[r.x, x + r.y, y + r.z, n]
    
    return {"output": output}
