# -----------------------------------------------------------------------------
# From Numpy to Python
# Copyright (2017) Nicolas P. Rougier - BSD license
# More information at https://github.com/rougier/numpy-book
# -----------------------------------------------------------------------------

import numpy as np
import dace as dc

XN, YN, M, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'M', 'N'])


@dc.program
def mgrid(X: dc.int64[M, N], Y: dc.int64[M, N]):
    for i in range(M):
        X[i, :] = i
    for j in range(N):
        Y[:, j] = j


@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dace.map[0:N]:
        X[i] = start + i * dist


@dc.program
def mandelbrot(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64,
               ymax: dc.float64, maxiter: dc.int64, horizon: dc.float64):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi = np.ndarray((XN, YN), dtype=np.int64)
    Yi = np.ndarray((XN, YN), dtype=np.int64)
    mgrid(Xi, Yi)
    X = np.ndarray((XN, ), dtype=np.float64)
    Y = np.ndarray((YN, ), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    # C = np.reshape(X, (xn, 1)) + Y * 1j
    C = np.ndarray((XN, YN), dtype=np.complex128)
    for i, j in dc.map[0:XN, 0:YN]:
        C[i, j] = X[i] + Y[j] * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    # Xiv = np.ndarray((XN * YN,), dtype=np.int64)
    Xiv = np.reshape(Xi, (XN * YN, ))
    # Yiv = np.ndarray((XN * YN,), dtype=np.int64)
    Yiv = np.reshape(Yi, (XN * YN, ))
    # Cv = np.ndarray((XN * YN,), dtype=np.complex128)
    Cv = np.reshape(C, (XN * YN, ))

    Z = np.zeros(Cv.shape, np.complex128)
    I = np.ndarray((XN * YN, ), dtype=np.bool_)
    length = XN * YN
    k = 0
    while length > 0 and k < maxiter:
        # for k in range(maxiter):
        #     if length <= 0:
        #         break

        # Compute for relevant points only
        Z[:length] = np.multiply(Z[:length], Z[:length])
        Z[:length] = np.add(Z[:length], Cv[:length])

        # Failed convergence
        I[:length] = np.absolute(Z[:length]) > horizon
        for j in range(length):
            if I[j]:
                N_[Xiv[j], Yiv[j]] = k + 1
        for j in range(length):
            if I[j]:
                Z_[Xiv[j], Yiv[j]] = Z[j]

        # Keep going with those who have not diverged yet
        I[:length] = np.logical_not(
            I[:length])  # np.negative(I, I) not working any longer
        count = 0
        # for j in range(length):
        #     if I[j]:
        #         Z[count] = Z[j]
        #         count += 1
        # count = 0
        # for j in range(length):
        #     if I[j]:
        #         Xiv[count] = Xiv[j]
        #         Yiv[count] = Yiv[j]
        #         count += 1
        # count = 0
        # for j in range(length):
        #     if I[j]:
        #         Cv[count] = Cv[j]
        #         count += 1
        for j in range(length):
            if I[j]:
                Z[count] = Z[j]
                Xiv[count] = Xiv[j]
                Yiv[count] = Yiv[j]
                Cv[count] = Cv[j]
                count += 1
        length = count
        k += 1
    return Z_.T, N_.T
