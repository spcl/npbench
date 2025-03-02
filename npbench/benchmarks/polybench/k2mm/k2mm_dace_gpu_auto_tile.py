import numpy as np
import dace as dc

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64)
                  for s in ('NI', 'NJ', 'NK', 'NL'))


@dc.program
def _kernel(alpha: dc.float64, beta: dc.float64, A: dc.float64[NI, NK],
           B: dc.float64[NK, NJ], C: dc.float64[NJ, NL], D: dc.float64[NI,
                                                                       NL]):

    D[:] = alpha * A @ B @ C + beta * D

_best_config = None

def autotuner(alpha, beta, A, B, C, D):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"alpha": alpha, "beta": beta, "A": A, "B": B, "C": C, "D": D},
        dims=get_max_ndim([alpha, beta, A, B, C, D])
    )

def kernel(alpha, beta, A, B, C, D):
    global _best_config
    _best_config(alpha, beta, A, B, C, D)
    return D
