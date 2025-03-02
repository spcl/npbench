import numpy as np
import dace as dc

M, N = (dc.symbol(s, dtype=dc.int64) for s in ('M', 'N'))


@dc.program
def _kernel(A: dc.float64[N, M], p: dc.float64[M], r: dc.float64[N]):

    return r @ A, A @ p

_best_config = None

def autotuner(M, N, A, p, r):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A, "p": p, "r": r, "M": M, "N":N},
        dims=get_max_ndim([A, p, r])
    )

def kernel(M, N, A, p, r):
    global _best_config
    R1, R2 = _best_config(M=M, N=N, A=A, p=p, r=r)
    return R1, R2
