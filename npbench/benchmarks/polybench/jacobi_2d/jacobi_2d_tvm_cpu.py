
import tvm
from tvm import te
from tvm import auto_scheduler
import tvm.testing
from tvm import autotvm
from npbench.infrastructure.tvm_cpu_framework import TVMCPUFramework

@auto_scheduler.register_workload("jacobi_2d_1_cpu")
def jacobi_2d_1_cpu(N, dtype):
    A = te.placeholder((N, N), name="A", dtype=dtype)

    def compute_step(A):
        return te.compute(
            (N, N),
            lambda i, j:
            te.if_then_else(
                te.all(i >= 1, i < N-1, j >= 1, j < N-1),
                0.2 * (
                    A[i, j] +   # center
                    A[i, j - 1] + # left
                    A[i, j + 1] + # right
                    A[i - 1, j] + # top
                    A[i + 1, j]   # bottom
                ),
                A[i, j]
            ),
            name="B_comp"
        )

    B_comp = compute_step(A)

    return [A, B_comp]

@auto_scheduler.register_workload("jacobi_2d_2_cpu")
def jacobi_2d_2_cpu(N, dtype):
    B = te.placeholder((N, N), name="B", dtype=dtype)

    def compute_step(B):
        return te.compute(
            (N, N),
            lambda i, j:
            te.if_then_else(
                te.all(i >= 1, i < N-1, j >= 1, j < N-1),
                0.2 * (
                    B[i, j] +   # center
                    B[i, j - 1] + # left
                    B[i, j + 1] + # right
                    B[i - 1, j] + # top
                    B[i + 1, j]   # bottom
                ),
                B[i, j]
            ),
            name="B_comp"
        )
    A_comp = compute_step(B)

    return [A_comp, B]

_kernel1 = None
_kernel2 = None

def autotuner(TSTEPS, A, B):
    global _kernel1
    global _kernel2

    if _kernel1 is not None and _kernel2 is not None:
        return

    dtype = A.dtype
    M = int(A.shape[0])
    N = int(A.shape[1])
    assert M == N

    _kernel1 = TVMCPUFramework.autotune(func=jacobi_2d_1_cpu, name="jacobi_2d_1_cpu", args=(N, dtype), target=tvm.target.Target("llvm"))
    _kernel2 = TVMCPUFramework.autotune(func=jacobi_2d_2_cpu, name="jacobi_2d_2_cpu", args=(N, dtype), target=tvm.target.Target("llvm"))



def kernel(TSTEPS, A, B):
    global _kernel1
    global _kernel2

    for _ in range(1, TSTEPS):
        _kernel1(A, B)
        _kernel2(A, B)
    return A