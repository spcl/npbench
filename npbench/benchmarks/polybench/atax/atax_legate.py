import legate.numpy as np
import timeit


def kernel(A, x):

    return (A @ x) @ A


def init_data(M, N, datatype):

    fn = datatype(N)
    A = np.empty((M, N), dtype=datatype)
    x = np.empty((N, ), dtype=datatype)
    y = np.empty((N, ), dtype=datatype)
    tmp = np.empty((M, ), dtype=datatype)
    # for i in range(N):
    #     x[i] = 1 + (i / fn)
    # for i in range(M):
    #     for j in range(N):
    #         A[i, j] = ((i + j) % N) / (5 * M)
    x[:] = np.random.randn(N)
    A[:] = np.random.randn(M, N)

    return A, x, y, tmp


if __name__ == "__main__":

    # Initialization
    M, N = 2000, 1000
    A, x, y, tmp = init_data(M, N, np.float64)

    # First execution
    lg_y = kernel(A, x)

    # Benchmark
    time = timeit.repeat("kernel(A, x)",
                         setup="pass",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(np.median(time)))
