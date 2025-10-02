import numpy as np

def kernel(NR, NQ, NP, A, C4):
    A[:] = (A.reshape(NR, NQ, 1, NP) @ C4).reshape(NR, NQ, NP)
    return (NR, NQ, NP, A, C4)