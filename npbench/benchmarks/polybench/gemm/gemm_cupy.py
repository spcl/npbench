import cupy as np

def kernel(alpha, beta, C, A, B):
    with np.cuda.Device(0):
        C[:] = alpha * A @ B + beta * C
    return (alpha, beta, C, A, B)