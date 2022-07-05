# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import numpy as np
from mpi4py import MPI


def azimint_naive(data, radius, npt):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    return res


def azimint_naive_easy(data, radius, npt):
    """ Distributes by `npt`. `data` and `radius` are replicated to all processes. """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = rank * int(np.ceil(npt / size))
    finish = min(npt, (rank + 1) * int(np.ceil(npt / size)))
    local_npt = finish - start

    if local_npt <= 0:
        return None

    rmax = radius.max()
    res = np.zeros(local_npt, dtype=np.float64)
    for i in range(start, finish):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i - start] = values_r12.mean()
    return res


def azimint_naive_hard(data, radius, npt, N):
    """ Distributes by `N`. `data` and `radius` are distributed accordingly. """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rmax = radius.max()  # Local reduction
    comm.AllReduce(MPI.IN_PLACE, rmax, MPI.MAX)  # Global reduction
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        # res[i] = values_r12.mean()
        num = values_r12.shape[0]
        comm.AllReduce(MPI.IN_PLACE, num, MPI.SUM)
        res[i] = values_r12.sum()
        comm.AllReduce(MPI.IN_PLACE, res[i], MPI.SUM)
        res[i] /= num
    return res
