import dpnp as np

def kernel(alpha, beta, A, B, C, D):
    D = np.add(np.multiply(alpha, np.dot(np.dot(A, B), C)), np.multiply(beta, D))
    return D
