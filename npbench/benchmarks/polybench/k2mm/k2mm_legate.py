import legate.numpy as np
import timeit

def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D
    return (alpha, beta, A, B, C, D)

def init_data(NI, NJ, NK, NL, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    tmp = np.empty((NI, NJ), dtype=datatype)
    A = np.empty((NI, NK), dtype=datatype)
    B = np.empty((NK, NJ), dtype=datatype)
    C = np.empty((NJ, NL), dtype=datatype)
    D = np.empty((NI, NL), dtype=datatype)
    A[:] = np.random.randn(NI, NK)
    B[:] = np.random.randn(NK, NJ)
    C[:] = np.random.randn(NJ, NL)
    D[:] = np.random.randn(NI, NL)
    return (alpha, beta, tmp, A, B, C, D)
if __name__ == '__main__':
    NI, NJ, NK, NL = (1000, 1000, 1000, 1000)
    alpha, beta, tmp, A, B, C, D = init_data(NI, NJ, NK, NL, np.float64)
    lg_D = np.copy(D)
    kernel(alpha, beta, A, B, C, lg_D)
    time = timeit.repeat('kernel(alpha, beta, A, B, C, lg_D)', setup='pass', repeat=20, number=1, globals=globals())
    print('Legate median time: {}'.format(np.median(time)))