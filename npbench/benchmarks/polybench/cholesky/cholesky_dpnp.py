import dpnp as np

def kernel(A):
    # Initial setup
#    A[0, 0] = np.sqrt(A[0, 0])
#    
#    for i in range(1, A.shape[0]):
#        for j in range(i):
#            A[i, j] -= np.dot(A[i, :j], A[j, :j])
#            A[i, j] /= A[j, j]
#        A[i, i] -= np.dot(A[i, :i], A[i, :i])
#        A[i, i] = np.sqrt(A[i, i])
    return np.linalg.cholesky(A)
