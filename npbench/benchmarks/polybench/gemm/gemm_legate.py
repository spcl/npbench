import argparse
import pathlib
import legate.numpy as np
from npbench import run, str2bool


def kernel(alpha, beta, C, A, B):

    C[:] = alpha * A @ B + beta * C


def init_data(NI, NJ, NK, datatype):

    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.empty((NI, NJ), dtype=datatype)
    # for i in range(NI):
    #     for j in range(NJ):
    #         C[i, j] = ((i * j + 1) % NI) / NI
    C[:] = np.random.randn(NI, NJ)
    A = np.empty((NI, NK), dtype=datatype)
    # for i in range(NI):
    #     for k in range(NK):
    #         A[i, k] = (i * (k + 1) % NK) / NK
    A[:] = np.random.randn(NI, NK)
    B = np.empty((NK, NJ), dtype=datatype)
    # for k in range(NK):
    #     for j in range(NJ):
    #         C[i, j] = (k * (j + 2) % NJ) / NJ
    B[:] = np.random.randn(NJ, NK)

    return alpha, beta, C, A, B


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="numpy")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a",
                        "--append",
                        type=str2bool,
                        nargs="?",
                        default=False)
    args = vars(parser.parse_args())

    print("Hello from Legate!")
    exit()

    # Initialization
    NI, NJ, NK = 2048, 2048, 2048
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)
    lg_C = orig_np.copy(C)

    # First execution
    kernel(alpha, beta, lg_C, A, B)

    # PAPI counters
    events = [papi_events.PAPI_DP_OPS, papi_events.PAPI_VEC_DP]
    papi_high.start_counters(events)
    lg_counters = [0] * len(events)

    def update_counters(old, new):
        for i in range(len(old)):
            old[i] += new[i]

    # Benchmark
    time = timeit.repeat(
        "papi_high.read_counters();"
        "kernel(alpha, beta, lg_C, A, B);"
        "update_counters(lg_counters, papi_high.read_counters());",
        setup="lg_C = orig_np.copy(C);",
        repeat=20,
        number=1,
        globals=globals())
    print("Numpy median time: {} ({} DP FLOPS)".format(
        np.median(time), (lg_counters[0] / 20) / np.median(time)))
