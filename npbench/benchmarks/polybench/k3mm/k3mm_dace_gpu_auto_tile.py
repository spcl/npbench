import numpy as np
import dace as dc

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64)
                      for s in ('NI', 'NJ', 'NK', 'NL', 'NM'))


@dc.program
def _kernel(A: dc.float64[NI, NK], B: dc.float64[NK, NJ], C: dc.float64[NJ, NM],
           D: dc.float64[NM, NL]):

    return A @ B @ C @ D

_best_config = None

def autotuner(A, B, C, D):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"A": A, "B": B, "C": C, "D": D},
        dims=get_max_ndim([A, B, C, D])
    )

def kernel(A, B, C, D):
    global _best_config
    E = _best_config(A, B, C, D)
    return E
