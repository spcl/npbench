import numpy as np
import dace as dc
import sympy as sp

# N, R, K, M1, M2 = (dc.symbol(s) for s in ('N', 'R', 'K', 'M1', 'M2'))
R, K, M1, M2 = (dc.symbol(s, dtype=dc.int64, integer=True, positive=True)
                for s in ('R', 'K', 'M1', 'M2'))
N = R**K

# TODO: Temporary fix!


@dc.program
def mgrid1(X: dc.uint32[R, R], Y: dc.uint32[R, R]):
    for i in range(R):
        X[i, :] = i
    for j in range(R):
        Y[:, j] = j


@dc.program
def mgrid2(X: dc.uint32[R, N], Y: dc.uint32[R, N]):
    for i in range(R):
        X[i, :] = i
    for j in range(R**K):
        Y[:, j] = j


@dc.program
def _stockham_fft(x: dc.complex128[R**K], y: dc.complex128[R**K]):

    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    # i_coord, j_coord = np.mgrid[0:R, 0:R]
    i_coord = np.ndarray((R, R), dtype=np.uint32)
    j_coord = np.ndarray((R, R), dtype=np.uint32)
    mgrid1(i_coord, j_coord)
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat[:] = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
    # Move input x to output y
    # to avoid overwriting the input.
    y[:] = x[:]

    # ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
    ii_coord = np.ndarray((R, N), dtype=np.uint32)
    jj_coord = np.ndarray((R, N), dtype=np.uint32)
    mgrid2(ii_coord, jj_coord)

    tmp_perm = np.empty_like(y)
    D = np.empty_like(y)
    tmp = np.empty_like(y)

    # Main Stockham loop
    for i in range(K):

        # Stride permutation
        yv = np.reshape(y, (R**i, R, R**(K - i - 1)))
        # tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        tmp_perm[:] = np.reshape(np.transpose(yv, axes=(1, 0, 2)), (N, ))
        # Twiddle Factor multiplication
        # D = np.empty((R, R ** i, R ** (K-i-1)), dtype=np.complex128)
        Dv = np.reshape(D, (R, R**i, R**(K - i - 1)))
        tmpv = np.reshape(tmp, (R**(K - i - 1), R, R**i))
        tmpv[0] = np.exp(-2.0j * np.pi * ii_coord[:, :R**i] *
                         jj_coord[:, :R**i] / R**(i + 1))
        for k in range(R**(K - i - 1)):
            # D[:, :, k] = tmp
            Dv[:, :, k] = np.reshape(tmpv[0], (R, R**i, 1))
        # tmp_twid = np.reshape(tmp_perm, (N, )) * np.reshape(D, (N, ))
        tmp_twid = tmp_perm * D
        # Product with Butterfly
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K - 1))),
                          (N, ))

_best_config = None

def autotuner(x, y):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _stockham_fft.to_sdfg(),
        {"x": x, "y": y},
        dims=get_max_ndim([x, y])
    )

def stockham_fft(x, y):
    global _best_config
    _best_config(x, y)
    return y
