import numpy as np

# Solves A @ x = b where A is a Compressed Sparse Row matrix using the Biconjugate Gradient Stabilized method
def bicgstab(A, b, x, max_iter=100, tol=np.float64(1e-6)):
	n = A.shape[0]
	r = b - A @ x
	rho_prev = alpha = omega = 1.0
	p = v = np.zeros_like(b)
	r_tilde = np.copy(r)
	x = np.copy(x)
	for i in range(max_iter):
		rho = np.dot(r_tilde.T, r)
		beta = (rho / rho_prev) * (alpha / omega)
		p = r + beta * (p - omega * v)
		v = A @ p
		alpha = rho / np.dot(r_tilde.T, v)
		s = r - alpha * v
		t = A @ s
		omega = np.dot(t.T, s) / np.dot(t.T, t)
		x = x + alpha * p + omega * s
		r = s - omega * t
		if np.linalg.norm(r) < tol:
			break
		rho_prev = rho
	return x
