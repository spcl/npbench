import legate.numpy as np
import time
import datetime

import math


def time_to_ms(raw):
    return int(round(raw * 1000))
    # return int(round(raw.total_seconds() * 1000))


def legate_benchmark(func, args, out_text="Legate", repeat=1):
    # NI, NJ, NK, NL, NM = 1600, 1800, 2000, 2200, 2400
    NI, NJ, NK, NL, NM = 2000, 2000, 2000, 2000, 2000
    E, A, B, F, C, D, G = init_data(NI, NJ, NK, NL, NM, np.float64)
    time_list = []
    for _ in range(max(1, repeat)):
        # start = datetime.datetime.now()
        start = time.time()
        # out = func(*args)
        # out = func(A, B, C, D)
        out = np.dot(np.dot(A, B), np.dot(C, D))
        # A, B, C, D = B, C, D, A
        # finish = datetime.datetime.now()
        # print(out[0, 0])
        assert not math.isnan(np.sum(out))
        # print("hello")
        finish = time.time()
        # print(start, finish)
        # print(out[0, 0])
        raw_time = finish - start
        time_list.append(raw_time)
    # time_list = [d.total_seconds() for d in time_list]
    # print(time_list)
    median_raw = np.median(time_list)
    ms_time = time_to_ms(median_raw)
    print("{}: {}ms".format(out_text, ms_time))
    return ms_time


def kernel(A, B, C, D):

    # return A @ B @ C @ D
    return np.dot(np.dot(A, B), np.dot(C, D))


def init_data(NI, NJ, NK, NL, NM, datatype):

    E = np.empty((NI, NJ), dtype=datatype)
    A = np.empty((NI, NK), dtype=datatype)
    B = np.empty((NK, NJ), dtype=datatype)
    F = np.empty((NJ, NL), dtype=datatype)
    C = np.empty((NJ, NM), dtype=datatype)
    D = np.empty((NM, NL), dtype=datatype)
    G = np.empty((NI, NL), dtype=datatype)
    # for i in range(NI):
    #     for j in range(NK):
    #         A[i, j] = ((i * j + 1) % NI) / (5 * NI)
    # for i in range(NK):
    #     for j in range(NJ):
    #         B[i, j] = ((i * (j + 1) + 2) % NJ) / (5 * NJ)
    # for i in range(NJ):
    #     for j in range(NM):
    #         C[i, j] = (i * (j + 3) % NL) / (5 * NL)
    # for i in range(NM):
    #     for j in range(NL):
    #         D[i, j] = ((i * (j + 2) + 2) % NK) / ( 5 * NK)
    A[:] = np.random.randn(NI, NK)
    B[:] = np.random.randn(NK, NJ)
    C[:] = np.random.randn(NJ, NM)
    D[:] = np.random.randn(NM, NL)
    # A = np.ones((NI, NK), datatype)
    # B = np.ones((NK, NJ), datatype)
    # C = np.ones((NJ, NM), datatype)
    # D = np.ones((NM, NL), datatype)

    return E, A, B, F, C, D, G


if __name__ == "__main__":

    # Initialization
    # NI, NJ, NK, NL, NM = 1600, 1800, 2000, 2200, 2400
    # E, A, B, F, C, D, G = init_data(NI, NJ, NK, NL, NM , np.float64)

    # First execution
    # G, _ = benchmark("kernel(A, B, C, D)",
    #  out_text="Legate first execution", context=globals())
    legate_benchmark(kernel, None, out_text="Legate first execution")

    # run_benchmark(legate_benchmark, 1, "3mm", (kernel, (A, B, C, D), 1, "Legate first execution"))

    # Benchmark
    # benchmark("kernel(A, B, C, D)",
    #           out_text="Legate median time",
    #           repeat=10, context=globals())
    legate_benchmark(kernel, None, out_text="Legate median time", repeat=10)
