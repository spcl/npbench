import numpy as np

# Solves A @ x = b where A is a Compressed Sparse Row matrix using the Conjugate Gradient method
def cg(A, x, b, max_iter=100, tol=np.float64(1e-6)):
	r = b - A @ x
	p = r
	rsold = r @ r
	for i in range(max_iter):
		Ap = A @ p
		alpha = rsold / (p @ Ap)
		x = x + alpha * p
		r = r - alpha * Ap
		rsnew = r @ r
		if np.sqrt(rsnew) < tol:
			break
		p = r + (rsnew / rsold) * p
		rsold = rsnew
	return x
