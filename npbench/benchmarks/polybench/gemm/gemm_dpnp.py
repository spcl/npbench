import dpnp as np

def kernel(alpha, beta, C, A, B):
    # Perform the matrix multiplication and addition
    C[:] = alpha * np.dot(A, B) + beta * C
