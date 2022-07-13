import argparse
import numpy as np
import sys
import timeit
from mpi4py import MPI


def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C


def initialize(start_M: int, start_N: int, start_K: int, tile_size_M: int, tile_size_N: int, tile_size_K: int, M, N, K, datatype: type = np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: (((i + start_M) * (j + start_N) + 1) % M) / M, (tile_size_M, tile_size_N), dtype=datatype)
    A = np.fromfunction(lambda i, k: ((i + start_M) * (k + start_K + 1) % K) / K, (tile_size_M, tile_size_K), dtype=datatype)
    B = np.fromfunction(lambda k, j: ((k + start_K) * (j + start_N + 2) % N) / N, (tile_size_K, tile_size_N), dtype=datatype)

    if start_M + tile_size_M > M:
        C[M - start_M:] = 0
        A[M - start_M:] = 0
    if start_N + tile_size_N > N:
        C[:, N - start_N:] = 0
        B[:, N - start_N:] = 0
    if start_K + tile_size_K > K:
        A[:, K - start_K:] = 0
        B[K - start_K:] = 0

    return alpha, beta, C, A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed GEMM kernel")
    parser.add_argument("-M", type=int, nargs="?", default=1024, metavar="A_rows", help="Number of A matrix rows")
    parser.add_argument("-N", type=int, nargs="?", default=1024, metavar="B_cols", help="Number of B matrix columns")
    parser.add_argument("-K", type=int, nargs="?", default=1024, metavar="A_cols/B_rows", help="Number of A (B) matrix columns (rows)")
    parser.add_argument("-PM", type=int, nargs="?", default=1, metavar="M_tiles", help="Number of A matrix row tiles")
    parser.add_argument("-PN", type=int, nargs="?", default=1, metavar="N_tiles", help="Number of B matrix column tiles")
    parser.add_argument("-PK", type=int, nargs="?", default=1, metavar="K_tiles", help="Number of A (B) matrix column (row) tiles")
    args = vars(parser.parse_args())

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    PM = args['PM']
    PN = args['PN']
    PK = args['PK']

    if any(tile < 1 for tile in (PM, PN, PK)):
        raise ValueError(f"All number of tiles {PM}, {PN}, and {PK} must be at least 1.")

    num_tiles = PM * PN * PK
    if num_tiles > world_size:
        raise ValueError(f"The total number of tiles {PM} * {PN} * {PK} = {num_tiles} cannot be greater than the total number of MPI processes {world_size}.")
    
    M = args['M']
    N = args['N']
    K = args['K']

    if any(size < tile for size, tile in zip((M, N, K), (PM, PN, PK))):
        raise ValueError(f"All matrix sizes {M}, {N}, and {K} must be at least equal to the corresponding number of tiles {PM}, {PN}, and {PK}.")

    cart_comm = world_comm.Create_cart([PM, PN, PK])

    if (cart_comm == MPI.COMM_NULL):
        world_comm.Barrier()
        sys.exit()

    cart_rank = cart_comm.Get_rank()
    cart_size = cart_comm.Get_size()
    cart_coords = cart_comm.Get_coords(cart_rank)

    tile_size_M = int(np.ceil(M / PM))
    tile_size_N = int(np.ceil(N / PN))
    tile_size_K = int(np.ceil(K / PK))

    start_M = cart_coords[0] * tile_size_M
    start_N = cart_coords[1] * tile_size_N
    start_K = cart_coords[2] * tile_size_K
    alpha, beta, C, A, B = initialize(start_M, start_N, start_K, tile_size_M, tile_size_N, tile_size_K, M, N, K)
    C_orig = C.copy()

    reduce_C_comm = cart_comm.Sub([False, True, False])

    def _func():
        kernel(alpha, beta, C, A, B)
        reduce_C_comm.Allreduce(MPI.IN_PLACE, C, op=MPI.SUM)
        cart_comm.Barrier()

    runtimes = timeit.repeat(
        stmt="_func()",
        setup="C[:] = C_orig; cart_comm.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if cart_rank == 0:
        print(f"Distributed GEMM kernels executed in {np.median(runtimes) * 1000} ms.")

    world_comm.Barrier()
