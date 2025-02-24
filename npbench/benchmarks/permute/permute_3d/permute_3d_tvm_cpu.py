import tvm
from tvm import te, autotvm, auto_scheduler
from npbench.infrastructure.tvm_framework import TVMFramework

@auto_scheduler.register_workload("permute3d_cpu")
def permute3d_cpu(N, dtype):
    A = te.placeholder((N, N, N), name="A", dtype=dtype)
    B = te.compute((N, N, N), lambda i, j, k: A[k, j, i], name="B")
    cfg = autotvm.get_config()
    cfg.add_flop(N**3)

    return [A, B]


def autotuner(A, B):
    global _kernel

    if _kernel is not None:
        return

    dtype = A.dtype
    M = int(A.shape[0])
    N = int(A.shape[1])
    K = int(A.shape[2])
    assert M == N and N == K
    _kernel = TVMFramework.autotune(func=permute3d_cpu, name="permute3d_cpu", args=(N, dtype), target=tvm.target.Target("llvm"))

_kernel = None

def kernel(A, B):
    global _kernel

    _kernel(A, B)

    return B