import numpy as np

# Solves A @ x = b where A is a Compressed Sparse Row matrix using the Biconjugate Gradient method
def bicg(A, b, x, max_iter=100, tol=np.float64(1e-6)):
	n = A.shape[0]
	r = b - A @ x
	r_tilde = np.copy(r)
	p = np.copy(r)
	p_tilde = np.copy(r_tilde)
	x = np.copy(x)
	rho = r_tilde.T @ r
	for _ in range(max_iter):
		Ap = A @ p
		alpha = rho / (p_tilde.T @ Ap)
		x = x + alpha * p
		r = r - alpha * Ap
		r_tilde = r_tilde - alpha * (A.T @ p_tilde)
		rho_new = r_tilde.T @ r
		beta = rho_new / rho
		p = r + beta * p
		p_tilde = r_tilde + beta * p_tilde
		if np.linalg.norm(r) < tol:
			break
		rho = rho_new
	return x
