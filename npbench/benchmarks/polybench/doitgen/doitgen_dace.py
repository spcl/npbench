import numpy as np
import dace as dc

NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NQ', 'NP'))


@dc.program
def kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):

    # Ideal - not working becayse Matmul with dim > 3 unsupported
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))
