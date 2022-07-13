import argparse
import numpy as np
import sys
import timeit
from mpi4py import MPI


def kernel(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


def distr_kernel(TSTEPS: int, A, B, cart_comm: MPI.Cartcomm):
    
    west, east = cart_comm.Shift(0, 1)
    woff, eoff = 1, A.shape[0] - 1 
    if west == MPI.PROC_NULL:
        woff += 1
    if east == MPI.PROC_NULL:
        eoff -= 1 
    wbuf = np.empty((1, ), dtype=A.dtype)
    ebuf = np.empty((1, ), dtype=A.dtype)
    req = np.empty((4, ), dtype=MPI.Request)

    for t in range(1, TSTEPS):

        req[0] = cart_comm.Isend(A[1], west, tag=0)
        req[1] = cart_comm.Isend(A[-2], east, tag=0)
        req[2] = cart_comm.Irecv(wbuf, west, tag=0)
        req[3] = cart_comm.Irecv(ebuf, east, tag=0)
        MPI.Request.Waitall(req)
        A[0] = wbuf
        A[-1] = ebuf

        B[woff:eoff] = 0.33333 * (A[woff-1:eoff-1] + A[woff:eoff] + A[woff+1:eoff+1])

        req[0] = cart_comm.Isend(B[1], west, tag=0)
        req[1] = cart_comm.Isend(B[-2], east, tag=0)
        req[2] = cart_comm.Irecv(wbuf, west, tag=0)
        req[3] = cart_comm.Irecv(ebuf, east, tag=0)
        MPI.Request.Waitall(req)
        B[0] = wbuf
        B[-1] = ebuf

        A[woff:eoff] = 0.33333 * (B[woff-1:eoff-1] + B[woff:eoff] + B[woff+1:eoff+1])



def initialize(start: int, tile_size: int, N, datatype=np.float64):
    A = np.fromfunction(lambda i: (i + start + 2) / N, (tile_size, ), dtype=datatype)
    B = np.fromfunction(lambda i: (i + start+ 3) / N, (tile_size, ), dtype=datatype)

    return A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed Jacobi-1D kernel")
    parser.add_argument("-N", type=int, nargs="?", default=1024, metavar="length", help="Length of vectors A and B")
    args = vars(parser.parse_args())

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    N = args['N']
    P = world_size
    if N < P:
        P = N

    cart_comm = world_comm.Create_cart([P])

    if (cart_comm == MPI.COMM_NULL):
        world_comm.Barrier()
        sys.exit()

    cart_rank = cart_comm.Get_rank()
    cart_size = cart_comm.Get_size()
    cart_coords = cart_comm.Get_coords(cart_rank)

    tile_size = int(np.ceil(N / P))

    start = cart_coords[0] * tile_size
    A, B = initialize(start, tile_size, N)
    A_padded = np.ndarray((tile_size + 2, ), dtype=A.dtype)
    B_padded = np.ndarray((tile_size + 2, ), dtype=B.dtype)
    A_padded[1:-1] = A
    B_padded[1:-1] = B
    A_orig = A_padded.copy()
    B_orig = B_padded.copy()

    def _func():
        distr_kernel(1000, A_padded, B_padded, cart_comm)
        cart_comm.Barrier()

    runtimes = timeit.repeat(
        stmt="_func()",
        setup="A_padded[:] = A_orig; B_padded[:] = B_orig; cart_comm.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if cart_rank == 0:
        print(f"Distributed Jacobi-1D kernel executed in {np.median(runtimes) * 1000} ms.")

    world_comm.Barrier()
