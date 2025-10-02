import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True, fastmath=True)
def mgrid(xn, yn):
    Xi = np.empty((xn, yn), dtype=np.uint32)
    Yi = np.empty((xn, yn), dtype=np.uint32)
    for i in range(xn):
        Xi[i, :] = i
    for j in range(yn):
        Yi[:, j] = j
    return (Xi, Yi)

@nb.jit(nopython=True, parallel=True, fastmath=True)
def stockham_fft(N, R, K, x, y):
    i_coord, j_coord = mgrid(R, R)
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord, jj_coord = mgrid(R, R ** K)
    for i in range(K):
        yv = np.reshape(y, (R ** i, R, R ** (K - i - 1)))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2)).copy()
        D = np.empty((R, R ** i, R ** (K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2j * np.pi * ii_coord[:, :R ** i] * jj_coord[:, :R ** i] / R ** (i + 1))
        for k in range(R ** (K - i - 1)):
            D[:, :, k] = tmp
        tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))
    return (N, R, K, x, y)