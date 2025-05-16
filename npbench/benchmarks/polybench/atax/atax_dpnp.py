import dpnp as np

def kernel(A, x):

    return np.dot(np.dot(A, x), A)
