import numpy as np
import dace as dc
N, H, SM = (dc.symbol(s, dc.int64) for s in ('N', 'H', 'SM'))

@dc.program
def _softmax(x: dc.float32[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    r = tmp_out / tmp_sum
    return r

@dc.program
def softmax_gpu(x: dc.float32[N, H, SM, SM], out: dc.float32[N, H, SM, SM]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum
    return (x, out)
_best_config = None

def autotuner(x):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, 'ndim')), default=0)
    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(_softmax.to_sdfg(), {'x': x}, dims=get_max_ndim([x]))

def softmax(x):
    global _best_config
    r = _best_config(x)
    return r