import numpy as np
import halide as hl

input_buffers = {
    "x": hl.ImageParam(hl.Float(32), 4, "x")
}

s4 = hl.Var()
s3 = hl.Var()
s2 = hl.Var()
s1 = hl.Var()

def softmax(x):
    a = hl.RDom([(0, x.width())])
    maxi = hl.Func("maxi")
    maxi[s3, s2, s1] = hl.maximum(x[a.x, s3, s2, s1])

    expo = hl.Func("expo")
    expo[s4, s3, s2, s1] = hl.exp(x[s4, s3, s2, s1] - maxi[s3, s2, s1])

    b = hl.RDom([(0, x.width())])
    nm = hl.Func("nm")
    nm[s3, s2, s1] = 0.0
    nm[s3, s2, s1] += expo[b.x, s3, s2, s1]

    output = hl.Func("output")
    output[s4, s3, s2, s1] = expo[s4, s3, s2, s1] / nm[s3, s2, s1]

    return {"output": output}

def set_estimates(x, output, N, H, SM):
    x.dim(0).set_estimate(0, SM)
    x.dim(1).set_estimate(0, SM)
    x.dim(2).set_estimate(0, H)
    x.dim(3).set_estimate(0, N)

    output.set_estimate(s4, 0, SM)
    output.set_estimate(s3, 0, SM)
    output.set_estimate(s2, 0, H)
    output.set_estimate(s1, 0, N)

