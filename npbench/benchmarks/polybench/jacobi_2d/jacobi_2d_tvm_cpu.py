
import tvm
from tvm import te
import tvm.testing
from tvm import autotvm
from npbench.infrastructure.tvm_cpu_framework import TVMCPUFramework

@autotvm.template("jacobi_2d_1")
def jacobi_2d_1(N, dtype):
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
    s = te.create_schedule(B_comp.op)


    cfg = autotvm.get_config()
    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_x", N, num_outputs=2)

    # Schedule B_comp
    b_y, b_x = s[B_comp].op.axis
    by1, ty1 = cfg["tile_y"].apply(s, B_comp, b_y)
    bx1, tx1 = cfg["tile_x"].apply(s, B_comp, b_x)

    # Reorder axes for B_comp
    s[B_comp].reorder(by1, bx1, ty1, tx1)

    return s, [A, B_comp]

@autotvm.template("jacobi_2d_2")
def jacobi_2d_2(N, dtype):
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
    s = te.create_schedule(A_comp.op)

    cfg = autotvm.get_config()
    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_x", N, num_outputs=2)

    # Schedule B_comp
    b_y, b_x = s[A_comp].op.axis
    by1, ty1 = cfg["tile_y"].apply(s, A_comp, b_y)
    bx1, tx1 = cfg["tile_x"].apply(s, A_comp, b_x)

    # Reorder axes for B_comp
    s[A_comp].reorder(by1, bx1, ty1, tx1)

    return s, [A_comp, B]


def autotuner(TSTEPS, A, B):
    global _kernel1
    global _kernel2

    if _kernel1 is not None and _kernel2 is not None:
        return

    dtype = A.dtype
    M = int(A.shape[0])
    N = int(A.shape[1])
    assert M == N
    target = tvm.target.Target("llvm")

    _kernel1 = TVMCPUFramework.autotune("jacobi_2d_1", __name__, (N, dtype), target)
    _kernel2 = TVMCPUFramework.autotune("jacobi_2d_2", __name__, (N, dtype), target)

_kernel1 = None
_kernel2 = None

def kernel(TSTEPS, A, B):
    global _kernel1
    global _kernel2

    for _ in range(1, TSTEPS):
        _kernel1(A, B)
        _kernel2(A, B)
    return A