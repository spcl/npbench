import legate.numpy as np
import timeit


def kernel(M, float_n, data):

    # mean = np.sum(data, axis=0) / float_n
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        for j in range(i, M):
            cov[i, j] = np.sum(data[:, i] * data[:, j])
            cov[i, j] /= float_n - 1.0
            cov[j, i] = cov[i, j]

    return cov


def init_data(M, N, datatype):

    float_n = datatype(N)
    data = np.empty((N, M), dtype=datatype)
    # for i in range(N):
    #     for j in range(M):
    #         data[i, j] = (i * j) / M + i
    data[:] = np.random.randn(N, M)

    return float_n, data


if __name__ == "__main__":

    # Initialization
    M, N = 100, 1000
    float_n, data = init_data(M, N, np.float64)
    lg_data = np.copy(data)

    # First execution
    kernel(M, float_n, lg_data)

    # Benchmark
    time = timeit.repeat("kernel(M, float_n, lg_data)",
                         setup="lg_data = np.copy(data)",
                         repeat=20,
                         number=1,
                         globals=globals())
    print("Legate median time: {}".format(np.median(time)))
