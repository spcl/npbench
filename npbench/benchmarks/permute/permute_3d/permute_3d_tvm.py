import tvm
from tvm import te, autotvm
from npbench.infrastructure.tvm_framework import TVMFramework

@autotvm.template("permute")
def permute(N, dtype):
    A = te.placeholder((N, N, N), name="A", dtype=dtype)
    B = te.compute((N, N, N), lambda i, j, k: A[k, j, i], name="B")

    s = te.create_schedule(B.op)
    cfg = autotvm.get_config()
    cfg.add_flop(N**3)

    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_x", N, num_outputs=2)
    cfg.define_split("tile_z", N, num_outputs=2)

    # Schedule B_comp
    b_z, b_y, b_x = s[B].op.axis
    bz1, tz1 = cfg["tile_z"].apply(s, B, b_z)
    by1, ty1 = cfg["tile_y"].apply(s, B, b_y)
    bx1, tx1 = cfg["tile_x"].apply(s, B, b_x)

    # Bind the threads for B_comp
    s[B].bind(bz1, te.thread_axis("blockIdx.z"))
    s[B].bind(by1, te.thread_axis("blockIdx.y"))
    s[B].bind(bx1, te.thread_axis("blockIdx.x"))
    s[B].bind(tz1, te.thread_axis("threadIdx.z"))
    s[B].bind(ty1, te.thread_axis("threadIdx.y"))
    s[B].bind(tx1, te.thread_axis("threadIdx.x"))

    return s, [A, B]


def autotuner(A, B):
    global _kernel

    if _kernel is not None:
        return

    dtype = A.dtype
    M = int(A.shape[0])
    N = int(A.shape[1])
    K = int(A.shape[2])
    assert M == N and N == K
    target = tvm.target.cuda()

    _kernel = TVMFramework.autotune("permute", __name__, (N, dtype), target)
    print("Permute kernel source:\n", _kernel.imported_modules[0].get_source())

_kernel = None

def kernel(A, B):
    global _kernel

    _kernel(A, B)

    return B