import dace
import numpy as np
from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework


N = dace.symbol("N")

@dace.program
def _kernel(A : dace.float64[N, N, N],
            B : dace.float64[N, N, N]):
    B = np.transpose(A, (2, 1, 0))


_best_config = None
__b = None

def autotuner(A, B, N):
    global _best_config
    global __b
    if _best_config is not None:
        return

    __best_config = DaceGPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B},
        dims=3
        )
    _best_config = __best_config.compile()
    args = {}
    kwargs = {"A": A, "B": B, "N": N}
    kwargs.update(
        # `_construct_args` will handle all of its arguments as kwargs.
        {aname: arg
            for aname, arg in zip(_best_config.argnames, args)})
    argtuple, initargtuple = _best_config._construct_args(kwargs)  # Missing arguments will be detected here.
    # Return values are cached in `self._lastargs`.
    #return self.fast_call(argtuple, initargtuple, do_gpu_check=True)
    __b = lambda: _best_config.fast_call(argtuple, initargtuple, False)



def kernel(A, B, N):
    global _best_config
    global __b
    #_best_config(A=A, B=B, N=N)
    __b()
    return B