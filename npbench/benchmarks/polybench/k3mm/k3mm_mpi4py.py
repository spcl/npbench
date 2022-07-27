import argparse
import numpy as np
import sys
import timeit
from mpi4py import MPI


def kernel(A, B, C, D):

    return A @ B @ C @ D


def initialize(b_NI: int, b_NJ: int, b_NK: int, b_NL: int, b_NM: int,
               ts_NI: int, ts_NJ: int, ts_NK, ts_NL: int, ts_NM: int,
               NI: int, NJ: int, NK: int, NL: int, NM: int,
               datatype: type = np.float64):

    A = np.fromfunction(lambda i, k: b_NK + k + 1, (ts_NI, ts_NK), dtype=datatype)
    B = np.eye(ts_NK, ts_NJ, b_NK - b_NJ)
    C = np.fromfunction(lambda j, m: b_NJ + j + 1, (ts_NJ, ts_NM), dtype=datatype)
    D = np.eye(ts_NM, ts_NL, b_NM - b_NL)

    if b_NI + ts_NI > NI:
        A[NI - b_NI:] = 0
    if b_NJ + ts_NJ > NJ:
        B[:, NJ - b_NJ:] = 0
        C[NJ - b_NJ:] = 0
    if b_NK + ts_NJ > NK:
        A[:, NK - b_NK:] = 0
        B[NK - b_NK:] = 0
    if b_NL + ts_NL > NL:
        D[:NL - b_NL] = 0
    if b_NM + ts_NM > NM:
        C[:NM - b_NM] = 0
        D[NM - b_NM:] = 0

    return A, B, C, D


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Distributed 3MM kernel")
    parser.add_argument("-NI", type=int, nargs="?", default=1024, metavar="A_rows", help="Number of A matrix rows")
    parser.add_argument("-NJ", type=int, nargs="?", default=1024, metavar="B_cols/C_rows", help="Number of B (C) matrix columns (rows")
    parser.add_argument("-NK", type=int, nargs="?", default=1024, metavar="A_cols/B_rows", help="Number of A (B) matrix columns (rows)")
    parser.add_argument("-NL", type=int, nargs="?", default=1024, metavar="C_cols/D_rows", help="Number of C (D) matrix columns (rows)")
    parser.add_argument("-NM", type=int, nargs="?", default=1024, metavar="D_cols", help="Number of D matrix cols")
    parser.add_argument("-PNI", type=int, nargs="?", default=1, metavar="NI_tiles", help="Number of A matrix row tiles")
    parser.add_argument("-PNJ", type=int, nargs="?", default=1, metavar="NJ_tiles", help="Number of B (C) matrix column (row) tiles")
    parser.add_argument("-PNK", type=int, nargs="?", default=1, metavar="NK_tiles", help="Number of A (B) matrix column (row) tiles")
    parser.add_argument("-PNL", type=int, nargs="?", default=1, metavar="NL_tiles", help="Number of C (D) matrix column (rows) tiles")
    parser.add_argument("-PNM", type=int, nargs="?", default=1, metavar="NM_tiles", help="Number of D matrix column tiles")
    args = vars(parser.parse_args())

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    PNI = args['PNI']
    PNJ = args['PNJ']
    PNK = args['PNK']
    PNL = args['PNL']
    PNM = args['PNM']

    if any(tile < 1 for tile in (PNI, PNJ, PNK, PNL, PNM)):
        raise ValueError(f"All number of tiles {PNI}, {PNJ}, {PNK}, {PNL}, and {PNM} must be at least 1.")

    num_tiles = PNI * PNJ * PNK
    if num_tiles > world_size:
        raise ValueError(f"The total number of tiles {PNI} * {PNJ} * {PNK} = {num_tiles} cannot be greater than the total number of MPI processes {world_size}.")
    num_tiles = PNM * PNJ * PNL
    if num_tiles > world_size:
        raise ValueError(f"The total number of tiles {PNM} * {PNJ} * {PNL} = {num_tiles} cannot be greater than the total number of MPI processes {world_size}.")
    num_tiles = PNI * PNJ * PNL
    if num_tiles > world_size:
        raise ValueError(f"The total number of tiles {PNI} * {PNJ} * {PNL} = {num_tiles} cannot be greater than the total number of MPI processes {world_size}.")
    if PNK != PNL:
        raise ValueError(f"Currently, only PNK ({PNK}) == PNL ({PNL}) value are supported.")
    if PNI != PNM:
        raise ValueError(f"Currently, only PNK ({PNI}) == PNL ({PNM}) value are supported.")


    NI = args['NI']
    NJ = args['NJ']
    NK = args['NK']
    NL = args['NL']
    NM = args['NM']

    if any(size < tile for size, tile in zip((NI, NJ, NK, NL, NM), (PNI, PNJ, PNK, PNL, PNM))):
        raise ValueError(f"All matrix sizes {NI}, {NJ}, {NK}, {NL}, and {NM} must be at least equal to the corresponding number of tiles {PNI}, {PNJ}, {PNK}, {PNL}, and {PNM}.")

    ab_cart_comm = world_comm.Create_cart([PNI, PNJ, PNK])
    # NOTE: Currently, PNK = PNL and PNI = PNM
    # cd_cart_comm = world_comm.Create_cart([PNM, PNJ, PNL])
    # abcd_cart_comm = world_comm.Create_cart([PNI, PNJ, PNL])

    if (ab_cart_comm == MPI.COMM_NULL):
        world_comm.Barrier()
        sys.exit()

    cart_rank = ab_cart_comm.Get_rank()
    cart_size = ab_cart_comm.Get_size()
    cart_coords = ab_cart_comm.Get_coords(cart_rank)

    ts_NI = int(np.ceil(NI / PNI))
    ts_NJ = int(np.ceil(NJ / PNJ))
    ts_NK = int(np.ceil(NJ / PNK))
    ts_NL = int(np.ceil(NL / PNL))
    ts_NM = int(np.ceil(NM / PNM))

    b_NI = cart_coords[0] * ts_NI
    b_NJ = cart_coords[1] * ts_NJ
    b_NK = cart_coords[2] * ts_NK
    b_NL = cart_coords[2] * ts_NL
    b_NM = cart_coords[0] * ts_NM
    A, B, C, D = initialize(b_NI, b_NJ, b_NK, b_NL, b_NM, ts_NI, ts_NJ, ts_NK, ts_NL, ts_NM, NI, NJ, NK, NL, NM)
    # D_orig = D.copy()

    ab_reduce_comm = ab_cart_comm.Sub([False, False, True])
    cd_reduce_comm = ab_cart_comm.Sub([True, False, False])
    abcd_reduce_comm = ab_cart_comm.Sub([False, True, False])

    def _func():
        ab = A @ B
        ab_reduce_comm.Allreduce(MPI.IN_PLACE, ab, op=MPI.SUM)
        cd = C @ D
        cd_reduce_comm.Allreduce(MPI.IN_PLACE, cd, op=MPI.SUM)
        E = ab @ cd
        abcd_reduce_comm.Allreduce(MPI.IN_PLACE, E, op=MPI.SUM)
        ab_cart_comm.Barrier()
        return E

    # Validate
    ab_cart_comm.Barrier()
    E = _func()
    val = NJ * (NJ + 1) * (2 * NJ + 1) / 6
    E_ref = np.fromfunction(lambda i, l: val, (ts_NI, ts_NL), dtype=np.float64)
    if b_NI + ts_NI > NI:
        E_ref[NI - ts_NI:] = 0
    if b_NL + ts_NL > NL:
        E_ref[:, NL - b_NL:] = 0
    assert(np.allclose(E, E_ref))
    ab_cart_comm.Barrier()

    runtimes = timeit.repeat(
        stmt="_func()",
        setup="ab_cart_comm.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if cart_rank == 0:
        print(f"Distributed 2MM kernel executed in {np.median(runtimes) * 1000} ms.")

    world_comm.Barrier()
