import numpy as np


# pythran export kernel(int, float64, float64[:, :])
def kernel(M, float_n, data):

    # mean = np.sum(data, axis=0) / float_n
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    # for i in range(M):
    #     for j in range(i, M):
    #         cov[i, j] = np.sum(data[:, i] * data[:, j])
    #         cov[i, j] /= float_n - 1.0
    #         cov[j, i] = cov[i, j]
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov
