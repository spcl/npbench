import numpy as np
import halide as hl

def softmax_params():
    x = hl.ImageParam(hl.Float(32), 4, "x")
    return (x,)

# Numerically-stable version of softmax
def softmax(x):
    S1 = 64
    S2 = 16
    S3 = 512
    S4 = 512

    s4 = hl.Var()
    s3 = hl.Var()
    s2 = hl.Var()
    s1 = hl.Var()

    a = hl.RDom([(0, S4)])
    maxi = hl.Func("maxi")
    maxi[s3, s2, s1] = hl.maximum(x[a.x, s3, s2, s1])

    expo = hl.Func("expo")
    expo[s4, s3, s2, s1] = hl.exp(x[s4, s3, s2, s1] - maxi[s3, s2, s1])

    b = hl.RDom([(0, S4)])
    nm = hl.Func("nm")
    nm[s3, s2, s1] = 0.0
    nm[s3, s2, s1] += expo[b.x, s3, s2, s1]

    output = hl.Func("output")
    output[s4, s3, s2, s1] = expo[s4, s3, s2, s1] / nm[s3, s2, s1]

    # Bounds

    x.dim(0).set_bounds(0, S4).set_stride(1)
    x.dim(1).set_bounds(0, S3).set_stride(S4)
    x.dim(2).set_bounds(0, S2).set_stride(S4 * S3)
    x.dim(3).set_bounds(0, S1).set_stride(S4 * S3 * S2)

    # Estimates

    x.dim(0).set_estimate(0, S4)
    x.dim(1).set_estimate(0, S3)
    x.dim(2).set_estimate(0, S2)
    x.dim(3).set_estimate(0, S1)

    output.set_estimate(s4, 0, S4)
    output.set_estimate(s3, 0, S3)
    output.set_estimate(s2, 0, S2)
    output.set_estimate(s1, 0, S1)

    return output
