import numpy as np
import dace as dc

_M, _N = (dc.symbol(s, dtype=dc.int64) for s in ('_M', '_N'))

@dc.program
def kernel(M, float_n: dc.float64, data: dc.float64[_N, _M]):
    return np.cov(np.transpose(data))


for i in 1..nbloks: # Host-side Loop
    # CUDA Kernel
    for k, j in cross(1..nlev, 1..nproma):
        if condition_1(k):
            kernel_body_1()
