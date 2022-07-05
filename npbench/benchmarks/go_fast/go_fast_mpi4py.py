# https://numba.readthedocs.io/en/stable/user/5minguide.html

import numpy as np
from mpi4py import MPI


def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


def go_fast_single(a, first_row):
    """ Distributes by `N`. By convention distributed by rows. """

    comm = MPI.COMM_WORLD

    trace = np.zeros((1,), dtype=a.dtype)
    for i in range(a.shape[0]):
        trace[0] += np.tanh(a[i, first_row + i])
    comm.Allreduce(MPI.IN_PLACE, trace, MPI.SUM)
    return a + trace


def go_fast_double(a, first_row, first_col):
    """ Distributes by `NxN`. """

    comm = MPI.COMM_WORLD

    trace = np.zeros((1,), dtype=a.dtype)
    for i in range(a.shape[0]):
        global_col = first_row + i
        local_col = global_col - first_col
        if local_col >= 0 and local_col < a.shape[1]:
            trace[0] += np.tanh(a[i, local_col])
    comm.Allreduce(MPI.IN_PLACE, trace, MPI.SUM)
    return a + trace


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 1000
    from go_fast import initialize
    a = initialize(N)

    ref = go_fast(a)

    start = rank * int(np.ceil(N / size))
    finish = min(N, (rank + 1) * int(np.ceil(N / size)))
    if finish - start > 0:
        single = go_fast_single(a[start:finish], start)
        assert(np.allclose(single, ref[start:finish]))
    

    Px = 2
    Py = size // 2
    P = Px * Py
    comm2 = comm.Create_cart([Px, Py])
    if rank < P:
        coords = comm2.Get_coords(rank)
        srow = coords[0] * int(np.ceil(N / Px))
        frow = min(N, (coords[0] + 1) * int(np.ceil(N / Px)))
        scol = coords[1] * int(np.ceil(N / Py))
        fcol = min(N, (coords[1] + 1) * int(np.ceil(N / Py)))
        if frow - srow > 0 and fcol - scol > 0:
            double = go_fast_double(a[srow:frow, scol:fcol], srow, scol)
            assert(np.allclose(double, ref[srow:frow, scol:fcol]))
