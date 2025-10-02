import numpy as np
import dace as dc
NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ('NR', 'NQ', 'NP'))

@dc.program
def _kernel(A: dc.float64[NR, NQ, NP], C4: dc.float64[NP, NP]):
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))
    return (A, C4)
_best_config = None

def autotuner(A, C4):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_kernel.to_sdfg(), {'A': A, 'C4': C4}, dims=get_max_ndim([A, C4]))

def kernel(A, C4):
    global _best_config
    _best_config(A, C4)
    return A