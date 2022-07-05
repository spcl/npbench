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


def azimint_naive_hard(data, radius, npt):
    """ Distributes by `N`. `data` and `radius` are distributed accordingly. """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rmax = np.empty((1,), dtype=radius.dtype)
    rmax[0] = radius.max()  # Local reduction
    comm.Allreduce(MPI.IN_PLACE, rmax, op=MPI.MAX)  # Global reduction
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        # res[i] = values_r12.mean()
        num = np.empty((1,), dtype=np.int32)
        num[0] = values_r12.shape[0]
        comm.Allreduce(MPI.IN_PLACE, num, MPI.SUM)
        tmp = np.empty((1,), dtype=values_r12.dtype)
        tmp[0] = values_r12.sum()
        comm.Allreduce(MPI.IN_PLACE, tmp, MPI.SUM)
        res[i] = tmp[0] / num
    return res


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 1000000
    npt = 10
    from azimint_naive import initialize
    data, radius = initialize(N)

    ref_res = azimint_naive(data, radius, npt)

    start = rank * int(np.ceil(npt / size))
    finish = min(npt, (rank + 1) * int(np.ceil(npt / size)))
    easy_res = azimint_naive_easy(data, radius, npt)
    if not (easy_res is None):
        assert(np.allclose(easy_res, ref_res[start:finish]))
    
    start = rank * int(np.ceil(N / size))
    finish = min(N, (rank + 1) * int(np.ceil(N / size)))
    if finish - start > 0:
        hard_res = azimint_naive_hard(data[start:finish], radius[start:finish], npt)
        assert(np.allclose(hard_res, ref_res))
