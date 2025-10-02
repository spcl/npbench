import numpy as np
import numba as nb

@nb.jit(nopython=False, forceobj=True, parallel=True, fastmath=True)
def stockham_fft(N, R, K, x, y):
    i_coord, j_coord = np.mgrid[0:R, 0:R]
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord, jj_coord = np.mgrid[0:R, 0:R ** K]
    for i in range(K):
        yv = np.reshape(y, (R ** i, R, R ** (K - i - 1)))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        D = np.empty((R, R ** i, R ** (K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2j * np.pi * ii_coord[:, :R ** i] * jj_coord[:, :R ** i] / R ** (i + 1))
        D[:] = np.repeat(np.reshape(tmp, (R, R ** i, 1)), R ** (K - i - 1), axis=2)
        tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))
    return (N, R, K, x, y)