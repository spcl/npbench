import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def _kernel(float_n: dc.float64, data: dc.float64[N, M]):

    mean = np.mean(data, axis=0)
    # data -= mean
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)
        cov[i:M, i] = cov[i, i:M]

    return cov

_best_config = None

def autotuner(float_n, data):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"float_n": float_n, "data": data},
        dims=get_max_ndim([float_n, data])
    )

def kernel(float_n, data):
    global _best_config
    cov = _best_config(float_n, data)
    return cov
