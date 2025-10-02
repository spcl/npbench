import numpy as np
import dace as dc
N = dc.symbol('N', dtype=dc.int64)

@dc.program
def _kernel(path: dc.int32[N, N]):
    for k in range(N):
        path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
    return path
_best_config = None

def autotuner(path, N):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'path': path, 'N': N}, dims=get_max_ndim([path]))

def kernel(path, N):
    global _best_config
    _best_config(path, N)
    return path