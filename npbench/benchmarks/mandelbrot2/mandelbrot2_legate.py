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


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    Xi, Yi = Xi.astype(np.uint32), Yi.astype(np.uint32)
    X = linspace(xmin, xmax, xn, dtype=np.float32)
    Y = linspace(ymin, ymax, yn, dtype=np.float32)
    X = np.repeat(X[:, None], yn, axis=1)
    Y = np.repeat(Y[None, :], xn, axis=0)
    N_ = np.zeros(X.shape, dtype=np.uint32)
    Zre_ = np.zeros(X.shape, dtype=np.float32)
    Zim_ = np.zeros(X.shape, dtype=np.float32)
    Xi = np.reshape(Xi, (xn * yn, ))
    Yi = np.reshape(Yi, (xn * yn, ))
    X = np.reshape(X, (xn * yn, ))
    Y = np.reshape(Y, (xn * yn, ))

    Zre = np.zeros(X.shape, np.float32)
    Zim = np.zeros(X.shape, np.float32)
    for i in range(itermax):
        if not len(Zre):
            break

        # Compute for relevant points only
        # np.multiply(Z, Z, Z)
        tmp = np.subtract(np.multiply(Zre, Zre), np.multiply(Zim, Zim))
        np.multiply(2, np.multiply(Zre, Zim), Zim)
        Zre[:] = tmp
        # np.add(Z, C, Z)
        Zre += X
        Zim += Y

        # Failed convergence
        I = np.sqrt(Zre**2 + Zim**2) > horizon
        for j in range(len(I)):
            if I[j]:
                N_[Xi[j], Yi[j]] = i + 1
                Zre_[Xi[j], Yi[j]] = Zre[j]
                Zim_[Xi[j], Yi[j]] = Zim[j]

        # Keep going with those who have not diverged yet
        np.logical_not(I, I)  # np.negative(I, I) not working any longer
        count = 0
        for j in range(len(I)):
            if I[j]:
                Zre[count] = Zre[j]
                Zim[count] = Zim[j]
                Xi[count] = Xi[j]
                Yi[count] = Yi[j]
                X[count] = X[j]
                Y[count] = Y[j]
                count += 1
        Zre = Zre[:count]
        Zim = Zim[:count]
        Xi = Xi[:count]
        Yi = Yi[:count]
        X = X[:count]
        Y = Y[:count]
    return Zre_.T, Zim_.T, N_.T
