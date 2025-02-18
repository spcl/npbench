import dace
import numpy as np

N = dace.symbol("N")

@dace.program
def kernel(A : dace.float64[N, N, N],
           B : dace.float64[N, N, N]):
    B = np.transpose(A, (2, 1, 0))
    return B