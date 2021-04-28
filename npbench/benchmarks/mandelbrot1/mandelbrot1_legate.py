# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import legate.numpy as np


def linspace(start, stop, num, dtype):
    X = np.empty((num, ), dtype=dtype)
    dist = (stop - start) / (num - 1)
    for i in range(num):
        X[i] = start + i * dist
    return X


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = linspace(xmin, xmax, xn, dtype=np.float32)
    Y = linspace(ymin, ymax, yn, dtype=np.float32)
    # C = X + Y[:,None]*1j
    N = np.zeros((xn, yn), dtype=np.int64)
    Zre = np.zeros((xn, yn), dtype=np.float32)
    Zim = np.zeros((xn, yn), dtype=np.float32)
    # for n in range(maxiter):
    #     I = np.less(np.sqrt(Zre**2 + Zim**2), horizon)
    #     N[:] = np.int64(I) * n + np.int64(~I) * N
    #     Zre[:] = np.int64(I) * (Zre**2 - Zim**2 + X[:, None]) + np.int64(~I)*Zre
    #     Zim[:] = np.int64(I) * (2 * Zre * Zim + Y[None, :]) + np.int64(~I)*Zim
    # I = np.not_equal(N, maxiter - 1)
    # N[:] = np.int64(I) * N
    for n in range(maxiter):
        I = np.less(np.sqrt(Zre**2 + Zim**2), horizon)
        N[I] = n
        # Z[I] = Z[I]**2 + C[I]
        tmp = Zre[I]**2 - Zim[I]**2 + X[I, None]
        Zim[I] = 2 * Zre[I] * Zim[I] + Y[None, I]
        Zre[I] = tmp
    N[N == maxiter - 1] = 0
    return Zre, Zim, N
