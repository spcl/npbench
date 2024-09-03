import dpnp as np

def kernel(A, p, r):
    return r @ A, A @ p
