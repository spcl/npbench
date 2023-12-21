import numpy as np

# Solves A @ x = b where A is a Compressed Sparse Row matrix using the Minimum Residual method
def hand_minres(A, b, x, max_iter=100, tol=1e-6):
	# Residual vector
	r = b - A @ x
	# Initial search direction
	p = r
	for _ in range(max_iter):
		Ap = A @ p
		alpha = (r @ r) / (p @ Ap)
		x = x + alpha * p
		r_new = r - alpha * Ap
		if np.linalg.norm(r_new) < tol:
			break
		beta = (r_new @ r_new) / (r @ r)
		p = r_new + beta * p
		r = r_new
	return x
