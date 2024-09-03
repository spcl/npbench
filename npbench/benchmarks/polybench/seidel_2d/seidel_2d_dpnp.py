import dpnp as np  

def kernel(TSTEPS, N, A):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (np.roll(A[i - 1, :-2], shift=1, axis=0) +
                           A[i - 1, 1:-1] +
                           np.roll(A[i - 1, 2:], shift=-1, axis=0) +
                           np.roll(A[i, 2:], shift=-1, axis=1) +
                           np.roll(A[i + 1, :-2], shift=1, axis=0) +
                           A[i + 1, 1:-1] +
                           np.roll(A[i + 1, 2:], shift=-1, axis=0))
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0
