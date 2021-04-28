import cupy as np


def kernel(A, x):

    return (A @ x) @ A
