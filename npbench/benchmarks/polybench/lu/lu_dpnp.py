import dpnp as np

def kernel(A):
    N = A.shape[0]
    for i in range(N):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[:j, j])
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= np.dot(A[i, :i], A[:i, j])
