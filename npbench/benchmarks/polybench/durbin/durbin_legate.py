import numpy as onp
import legate.numpy as np
import timeit

import durbin_numpy as np_impl


def kernel(r):

    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        # y[:k] += alpha * np.flip(y[:k])
        tmp = np.flip(y[:k])
        y[:k] += alpha * tmp
        y[k] = alpha

    return y


def init_data(N, datatype):

    r = onp.empty((N, ), dtype=datatype)
    y = onp.empty((N, ), dtype=datatype)
    for i in range(N):
        r[i] = N + 1 - i

    return r, y


if __name__ == "__main__":

    # Initialization
    N = 1000
    r, y = init_data(N, np.float64)

    # First execution
    np_y = np_impl.kernel(r)
    lg_y = kernel(r)

    # Validation
    assert (np.allclose(np_y, lg_y))

    # Benchmark
    time = timeit.repeat("np_impl.kernel(r)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Numpy median time: {}".format(np.median(time)))
    time = timeit.repeat("kernel(r)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(np.median(time)))
