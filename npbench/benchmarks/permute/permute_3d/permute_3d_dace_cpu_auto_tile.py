import dace
import numpy as np
from npbench.infrastructure.dace_cpu_auto_tile_framework import DaceCPUAutoTileFramework


N = dace.symbol("N")

@dace.program
def _kernel(A : dace.float64[N, N, N],
            B : dace.float64[N, N, N]):
    B = np.transpose(A, (2, 1, 0))


_best_config = None


def autotuner(A, B, N):
    global _best_config
    if _best_config is not None:
        return

    __best_config, _ = DaceCPUAutoTileFramework.autotune(
        _kernel.to_sdfg(),
        {"N": N, "A": A, "B": B},
        dims=3
        )
    _best_config = __best_config.compile()

def kernel(A, B, N):
    global _best_config
    _best_config(A=A, B=B, N=N)
    return B