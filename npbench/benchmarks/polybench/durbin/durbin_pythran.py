import numpy as np


def flip(A):
    B = np.empty_like(A)
    for i in range(B.shape[0]):
        B[i] = A[-1 - i]
    return B


# pythran export kernel(float64[:])
def kernel(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        # alpha = - (r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        alpha = -(r[k] + np.dot(flip(r[:k]), y[:k])) / beta
        # y[:k] += alpha * np.flip(y[:k])
        y[:k] += alpha * flip(y[:k])
        y[k] = alpha

    return y
