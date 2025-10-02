import numpy as np

def mgrid(xn, yn):
    Xi = np.empty((xn, yn), dtype=np.uint32)
    Yi = np.empty((xn, yn), dtype=np.uint32)
    for i in range(xn):
        Xi[i, :] = i
    for j in range(yn):
        Yi[:, j] = j
    return (Xi, Yi)

def stockham_fft(N, R, K, x, y):
    i_coord, j_coord = mgrid(R, R)
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord, jj_coord = mgrid(R, R ** K)
    for i in range(K):
        yv = y.reshape(R ** i, R, R ** (K - i - 1))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        D = np.empty((R, R ** i, R ** (K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2j * np.pi * ii_coord[:, :R ** i] * jj_coord[:, :R ** i] / R ** (i + 1))
        D[:] = np.repeat(tmp.reshape(R, R ** i, 1), R ** (K - i - 1), axis=2)
        tmp_twid = tmp_perm.reshape(N) * D.reshape(N)
        tmp2 = dft_mat @ tmp_twid.reshape(R, R ** (K - 1))
        y[:] = tmp2.reshape(N)
    return (N, R, K, x, y)