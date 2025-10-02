import numpy as np
import dace as dc
XN, YN, M, N = (dc.symbol(s, dtype=dc.int64) for s in ['XN', 'YN', 'M', 'N'])

@dc.program
def mgrid(X: dc.int64[M, N], Y: dc.int64[M, N]):
    for i in range(M):
        X[i, :] = i
    for j in range(N):
        Y[:, j] = j
    return (X, Y)

@dc.program
def linspace(start: dc.float64, stop: dc.float64, X: dc.float64[N]):
    dist = (stop - start) / (N - 1)
    for i in dace.map[0:N]:
        X[i] = start + i * dist
    return (start, stop, X)

@dc.program
def _mandelbrot(xmin: dc.float64, xmax: dc.float64, ymin: dc.float64, ymax: dc.float64, maxiter: dc.int64, horizon: dc.float64):
    Xi = np.ndarray((XN, YN), dtype=np.int64)
    Yi = np.ndarray((XN, YN), dtype=np.int64)
    mgrid(Xi, Yi)
    X = np.ndarray((XN,), dtype=np.float64)
    Y = np.ndarray((YN,), dtype=np.float64)
    linspace(xmin, xmax, X)
    linspace(ymin, ymax, Y)
    C = np.ndarray((XN, YN), dtype=np.complex128)
    for i, j in dc.map[0:XN, 0:YN]:
        C[i, j] = X[i] + Y[j] * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xiv = np.reshape(Xi, (XN * YN,))
    Yiv = np.reshape(Yi, (XN * YN,))
    Cv = np.reshape(C, (XN * YN,))
    Z = np.zeros(Cv.shape, np.complex128)
    I = np.ndarray((XN * YN,), dtype=np.bool_)
    length = XN * YN
    k = 0
    while length > 0 and k < maxiter:
        Z[:length] = np.multiply(Z[:length], Z[:length])
        Z[:length] = np.add(Z[:length], Cv[:length])
        I[:length] = np.absolute(Z[:length]) > horizon
        for j in range(length):
            if I[j]:
                N_[Xiv[j], Yiv[j]] = k + 1
        for j in range(length):
            if I[j]:
                Z_[Xiv[j], Yiv[j]] = Z[j]
        I[:length] = np.logical_not(I[:length])
        count = 0
        for j in range(length):
            if I[j]:
                Z[count] = Z[j]
                Xiv[count] = Xiv[j]
                Yiv[count] = Yiv[j]
                Cv[count] = Cv[j]
                count += 1
        length = count
        k += 1
    return (Z_.T, N_.T)
_best_config = None

def autotuner(xmin, xmax, ymin, ymax, maxiter, horizon):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_mandelbrot.to_sdfg(), {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'maxiter': maxiter, 'horizon': horizon}, dims=get_max_ndim([xmin, xmax, ymin, ymax, maxiter, horizon]))

def mandelbrot(xmin, xmax, ymin, ymax, maxiter, horizon):
    global _best_config
    ZT, NT = _best_config(xmin, xmax, ymin, ymax, maxiter, horizon)
    return (ZT, NT)