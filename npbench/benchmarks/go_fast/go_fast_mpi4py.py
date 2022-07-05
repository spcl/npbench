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
    rank = comm.Get_rank()
    size = comm.Get_size()

    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, first_row + i])
    comm.AllReduce(MPI.IN_PLACE, trace, MPI.SUM)
    return a + trace


def go_fast_double(a, first_row, first_col):
    """ Distributes by `NxN`. """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    trace = 0.0
    for i in range(a.shape[0]):
        global_col = first_row + i
        local_col = global_col - first_col
        if local_col >= 0 and local_col < a.shape[1]:
            trace += np.tanh(a[i, local_col])
    comm.AllReduce(MPI.IN_PLACE, trace, MPI.SUM)
    return a + trace
