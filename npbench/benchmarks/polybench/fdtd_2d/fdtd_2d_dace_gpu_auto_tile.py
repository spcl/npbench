import numpy as np
import dace as dc

TMAX, NX, NY = (dc.symbol(s, dtype=dc.int64) for s in ('TMAX', 'NX', 'NY'))


@dc.program
def _kernel(ex: dc.float64[NX, NY], ey: dc.float64[NX, NY],
           hz: dc.float64[NX, NY], _fict_: dc.float64[TMAX]):

    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])

_best_config = None

def autotuner(ex, ey, hz, _fict_):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"ex": ex, "ey": ey, "hz": hz, "_fict_": _fict_},
        dims=get_max_ndim([ex, ey, hz, _fict_])
    )

def kernel(ex, ey, hz, _fict_):
    global _best_config
    _best_config(ex, ey, hz, _fict_)
    return ex
