import dpnp as np

def kernel(alpha, beta, C, A, B):
    # Perform the matrix multiplication and addition
     C = np.add(np.multiply(alpha, np.dot(A, B)), np.multiply(beta, C))
     return C
