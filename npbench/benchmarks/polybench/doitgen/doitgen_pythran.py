import numpy as np


# pythran export kernel(int, int, int, float64[:,:,:], float64[:,:])
def kernel(NR, NQ, NP, A, C4):

    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    A[:] = (A.reshape(NR, NQ, 1, NP) @ C4).reshape(NR, NQ, NP)
