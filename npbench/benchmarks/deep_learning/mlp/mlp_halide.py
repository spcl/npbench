import numpy as np
import halide as hl

input_buffers = {
    "input": hl.ImageParam(hl.Float(32), 2, "input"),
    "w1": hl.ImageParam(hl.Float(32), 2, "w1"),
    "b1": hl.ImageParam(hl.Float(32), 1, "b1"),
    "w2": hl.ImageParam(hl.Float(32), 2, "w2"),
    "b2": hl.ImageParam(hl.Float(32), 1, "b2"),
    "w3": hl.ImageParam(hl.Float(32), 2, "w3"),
    "b3": hl.ImageParam(hl.Float(32), 1, "b3")
}

n = hl.Var("n")
h1 = hl.Var("h1")
h2 = hl.Var("h2")
h3 = hl.Var("h3")

def set_estimates(input, w1, b1, w2, b2, w3, b3, output, N, C_in, S0, S1, S2):
    input.dim(0).set_estimate(0, C_in)
    input.dim(1).set_estimate(0, N)

    w1.dim(0).set_estimate(0, C_in)
    w1.dim(1).set_estimate(0, S0)
    b1.dim(0).set_estimate(0, S0)

    w2.dim(0).set_estimate(0, S0)
    w2.dim(1).set_estimate(0, S1)
    b2.dim(0).set_estimate(0, S1)

    w3.dim(0).set_estimate(0, S1)
    w3.dim(1).set_estimate(0, S2)
    b3.dim(0).set_estimate(0, S2)

    output.set_estimate(h3, 0, S2)
    output.set_estimate(n, 0, N)

# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3):
    
    r1 = hl.RDom([(0, input.width())])
    layer1 = hl.Func("layer1")
    layer1[h1, n] = b1[h1]
    layer1[h1, n] += input[r1.x, n] * w1[r1.x, h1]

    relu1 = hl.Func("relu1")
    relu1[h1, n] = hl.max(0.0, layer1[h1, n])

    # Layer 2
    r2 = hl.RDom([(0, b1.width())])
    layer2 = hl.Func()
    layer2[h2, n] = b2[h2]
    layer2[h2, n] += layer1[r2.x, n] * w2[r2.x, h2]

    relu2 = hl.Func("relu2")
    relu2[h2, n] = hl.max(0.0, layer2[h2, n])

    # Layer 3
    r3 = hl.RDom([(0, b2.width())])
    layer3 = hl.Func("layer3")
    layer3[h3, n] = b3[h3]
    layer3[h3, n] += layer2[r3.x, n] * w3[r3.x, h3]

    # Softmax

    a = hl.RDom([(0, b3.width())])
    maxi = hl.Func("maxi")
    maxi[n] = hl.maximum(layer3[a.x, n])

    expo = hl.Func("expo")
    expo[h3, n] = hl.exp(layer3[h3, n] - maxi[n])

    b = hl.RDom([(0, b3.width())])
    norm = hl.Func("norm")
    norm[n] = 0.0
    norm[n] += expo[b.x, n]

    output = hl.Func("output")
    output[h3, n] = expo[h3, n] / norm[n]

    return {"output": output}
