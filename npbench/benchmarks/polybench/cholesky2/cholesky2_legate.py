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
    return A

def kernel2(A):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
    return A

def init_data(N, datatype):
    A = np.empty((N, N), dtype=datatype)
    A[:] = np.random.randn(N, N)
    A[:] = A @ np.transpose(A)
    return A
if __name__ == '__main__':
    N = 1000
    A = init_data(N, np.float64)
    lg_A = np.copy(A)
    lg_A2 = np.copy(A)
    kernel2(lg_A2)
    time = timeit.repeat('kernel2(lg_A2)', setup='lg_A2 = np.copy(A)', repeat=20, number=1, globals=globals())
    print('Legate (2nd kernel) median time: {}'.format(np.median(time)))