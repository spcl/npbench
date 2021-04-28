import legate.numpy as np
import timeit


def kernel(A, p, r):

    return r @ A, A @ p


def init_data(M, N, datatype):

    A = np.empty((N, M), dtype=datatype)
    s = np.empty((M, ), dtype=datatype)
    q = np.empty((N, ), dtype=datatype)
    p = np.empty((M, ), dtype=datatype)
    r = np.empty((N, ), dtype=datatype)
    # for i in range(M):
    #     p[i] = (i % M) / M
    # for i in range(N):
    #     r[i] = (i % N) / N
    #     for j in range(M):
    #         A[i, j] = (i * (j + 1) % N) / N
    p[:] = np.random.randn(M)
    r[:] = np.random.randn(N)
    A[:] = np.random.randn(N, M)

    return A, s, q, p, r


if __name__ == "__main__":

    # Initialization
    M, N = 2000, 1000
    A, s, q, p, r = init_data(M, N, np.float64)

    # First execution
    lg_s, lg_q = kernel(A, p, r)

    # Benchmark
    time = timeit.repeat("kernel(A, p, r)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(np.median(time)))
