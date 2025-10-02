import numpy as np
import dace as dc
N = dc.symbol('N', dtype=dc.int64)

@dc.program
def kernel(path: dc.int32[N, N]):
    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
    return path