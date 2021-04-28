import legate.numpy as np
import timeit


def kernel(A):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


def kernel2(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)


def init_data(N, datatype):

    A = np.empty((N, N), dtype=datatype)
    # for i in range(N):
    #     for j in range(i + 1):
    #         A[i, j] = (-j % N) / N + 1
    #     for j in range(i + 1, N):
    #         A[i, j] = 0.0
    #     A[i, i] = 1.0
    A[:] = np.random.randn(N, N)

    # B = np.empty((N, N), dtype=datatype)
    # for r in range(N):
    #     for s in range(N):
    #         B[r, s] = 0.0
    # for t in range(N):
    #     for r in range(N):
    #         for s in range(N):
    #             B[r, s] += A[r, t] * A[s, t]
    # for r in range(N):
    #     for s in range(N):
    #         A[r, s] = B[r, s]
    A[:] = A @ np.transpose(A)

    return A


if __name__ == "__main__":

    # Initialization
    N = 1000
    A = init_data(N, np.float64)
    lg_A = np.copy(A)
    lg_A2 = np.copy(A)

    # First execution
    # kernel(lg_A)
    kernel2(lg_A2)

    # Benchmark
    # time = timeit.repeat("kernel(lg_A)", setup="lg_A = np.copy(A)",
    #                      repeat=20, number=1, globals=globals())
    # print("Legate median time: {}".format(np.median(time)))
    time = timeit.repeat("kernel2(lg_A2)",
                         setup="lg_A2 = np.copy(A)",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate (2nd kernel) median time: {}".format(np.median(time)))
