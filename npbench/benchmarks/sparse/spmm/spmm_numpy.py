import numpy as np

# Stores the result of the product of 2 Compressed Sparse Row matrices A and B in C as a dense matrice
def spmm(alpha, beta, C, A, B):
	C[:] = alpha * A @ B + beta * C
