import dace
import numpy as np

N = dace.symbol("N")

@dace.program
def _kernel(A : dace.float64[N, N, N],
           B : dace.float64[N, N, N]):
    B = np.transpose(A, (2, 1, 0))
    return B

_best_config = None

def autotuner(A, B, N):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A, "B": B, "N": N},
        dims=get_max_ndim([A, B])
    )

def kernel(A, B, N):
    global _best_config
    _best_config(A=A, B=B, N=N)
    return B
