import numpy as np
import dace as dc
NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NQ', 'NP'))

@dc.program
def kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))
    return (A, C4)