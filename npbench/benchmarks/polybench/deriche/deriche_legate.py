import numpy as onp
import legate.numpy as np
import timeit

import deriche_numpy as np_impl


def kernel(alpha, imgIn):

    k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (
        1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0**(-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    # y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    y1[:, 1] = b1 * y1[:, 0]
    y1[:, 1] += a1 * imgIn[:, 1] + a2 * imgIn[:, 0]
    for j in range(2, imgIn.shape[1]):
        y1[:, j] = (a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] +
                    b1 * y1[:, j - 1] + b2 * y1[:, j - 2])

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(imgIn.shape[1] - 3, -1, -1):
        y2[:, j] = (a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] +
                    b1 * y2[:, j + 1] + b2 * y2[:, j + 2])

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, imgIn.shape[0]):
        y1[i, :] = (a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] +
                    b1 * y1[i - 1, :] + b2 * y1[i - 2, :])

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(imgIn.shape[0] - 3, -1, -1):
        y2[i, :] = (a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] +
                    b1 * y2[i + 1, :] + b2 * y2[i + 2, :])

    imgOut[:] = c2 * (y1 + y2)

    return imgOut


def init_data(W, H, datatype):

    alpha = datatype(0.25)
    imgIn = onp.empty((W, H), dtype=datatype)
    imgOut = onp.empty((W, H), dtype=datatype)
    y1 = onp.empty((W, H), dtype=datatype)
    y2 = onp.empty((W, H), dtype=datatype)
    for i in range(W):
        for j in range(H):
            imgIn[i, j] = ((313 * i + 991 * j) % 65536) / 65535.0

    return alpha, imgIn, imgOut, y1, y2


if __name__ == "__main__":

    # Initialization
    W, H = 1000, 1000
    alpha, imgIn, imgOut, y1, y2 = init_data(W, H, np.float64)

    # First execution
    np_imgOut = np_impl.kernel(alpha, imgIn)
    lg_imgOut = kernel(alpha, imgIn)

    # Validation
    assert (onp.allclose(np_imgOut, lg_imgOut))

    # Benchmark
    time = timeit.repeat("np_impl.kernel(alpha, imgIn)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("NumPy median time: {}".format(np.median(time)))
    time = timeit.repeat("kernel(alpha, imgIn)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(np.median(time)))
