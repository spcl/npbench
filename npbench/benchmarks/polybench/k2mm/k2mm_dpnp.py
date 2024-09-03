import dpnp as np

def kernel(alpha, beta, A, B, C, D):

    D[:] = alpha * np.dot(np.dot(A, B), C) + beta * D
