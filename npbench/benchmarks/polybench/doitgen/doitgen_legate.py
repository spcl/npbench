import numpy as onp
import legate.numpy as np
import timeit

import doitgen_numpy as np_impl


def kernel(NR, NQ, NP, A, C4):

    for r in range(NR):
        for q in range(NQ):
            tmp = A[r, q, :] @ C4
            A[r, q, :] = tmp
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


def init_data(NR, NQ, NP, datatype):

    A = onp.empty((NR, NQ, NP), dtype=datatype)
    C4 = onp.empty((
        NP,
        NP,
    ), dtype=datatype)
    sum = onp.empty((NP, ), dtype=datatype)
    for i in range(NR):
        for j in range(NQ):
            for k in range(NP):
                A[i, j, k] = ((i * j + k) % NP) / NP
    for i in range(NP):
        for j in range(NP):
            C4[i, j] = (i * j % NP) / NP

    return A, C4, sum


if __name__ == "__main__":

    # Initialization
    NR, NQ, NP = 10, 10, 1000
    A, C4, sum = init_data(NR, NQ, NP, np.float64)
    np_A = onp.copy(A)
    lg_A = onp.copy(A)

    # First execution
    np_impl.kernel(NR, NQ, NP, np_A, C4)
    kernel(NR, NQ, NP, lg_A, C4)

    # Validation
    assert (onp.allclose(np_A, lg_A))

    # Benchmark
    time = timeit.repeat("np_impl.kernel(NR, NQ, NP, np_A, C4)",
                         setup="np_A = onp.copy(A)",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Numpy median time: {}".format(onp.median(time)))
    time = timeit.repeat("kernel(NR, NQ, NP, lg_A, C4)",
                         setup="lg_A = onp.copy(A)",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(onp.median(time)))
