import dpnp as np

def kernel(A):
    return np.linalg.cholesky(A)
