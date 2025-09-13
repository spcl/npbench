# Bounded Matrix_1 * Matrix_2 * Transposed_1
import numpy as np

# Returns the transposed of a banded square matrix
def transposed(A : np.ndarray, lbound : int, ubound : int):
	size = A.shape[0]
	width = A.shape[1]
	ret = np.zeros([size, width])
	# Stores the indexes of the first non-zero element for
	# each line in the transposed matrix
	ret_start = np.array(list(map(lambda i: max(i - ubound, 0), range(0, size))))
	# for i in range(0, size):
	#	 ret_start[i] = max(i - ubound, 0)
	for i in range(0, size):
		start = max(i - lbound, 0)
		stop = min(size, i + ubound + 1)
		for j in range(0, stop - start):
			# Get the column of the current element in the coresponding dense matrix
			dense_j = j + start
			ret[dense_j][i - ret_start[dense_j]] = A[i][j]
	return ret


# Returns the product of a banded square matrix with the transposed of
# the another banded square matrix
# Also returns the bounds of the resulted matrix
def banded_dgemt(A : np.ndarray, a_lbound : int, a_ubound : int, B : np.ndarray, b_lbound : int, b_ubound : int):
	if not A.shape[0] == B.shape[0]:
		print(f"Cannot multiply square matrixes with different sizes {A.shape[0]} {B.shape[0]}")
		return None
	size = A.shape[0]
	# A bound cannot be less than -size or bigger than size - 1
	lbound = max(-size, min(a_lbound + b_ubound, size - 1))
	ubound = max(-size, min(a_ubound + b_lbound, size - 1))
	ret = np.zeros([size, min(size, 1 + lbound + ubound)])
	for i in range(0, size):
		start = max(i - lbound, 0)
		stop = min(size, i + ubound + 1)
		a_start = max(0, i - a_lbound)
		a_cnt = 1 + min(size - i - 1, a_ubound) + min(i, a_lbound)
		a_stop = min(size, i + a_ubound + 1)
		for j in range(start, stop):
			b_start = max(0, j - b_lbound)
			b_cnt = 1 + min(size - j - 1, b_ubound) + min(j, b_lbound)
			b_stop = min(size, i + b_ubound + 1)
			acc = A.dtype.type(0)
			offset_a = 0
			offset_b = 0
			if a_start >= b_start:
				offset_a = a_start - b_start
			else:
				offset_b = b_start - a_start
			interval = min(a_cnt - offset_b, b_cnt - offset_a)
			ret[i][j - start] = A[i][offset_b : offset_b + interval] @ B[j][offset_a : offset_a + interval]
	return ret, lbound, ubound

# Returns the product of a banded square matrix another banded square matrix
# Also returns the bounds of the resulted matrix
def banded_dgemm(A : np.ndarray, a_lbound : int, a_ubound : int, B : np.ndarray, b_lbound : int, b_ubound : int):
	if not A.shape[0] == B.shape[0]:
		print(f"Cannot multiply square matrixes with different sizes {A.shape[0]} {B.shape[0]}")
		return None
	return banded_dgemt(A, a_lbound, a_ubound, transposed(B, b_lbound, b_ubound), b_ubound, b_lbound)

# Returns the result of the A @ B @ A^T and its bounds
def banded_mmt(A : np.ndarray, a_lbound : int, a_ubound : int,
	B : np.ndarray, b_lbound : int, b_ubound : int):
	ret, lbound, ubound = banded_dgemm(A, a_lbound, a_ubound, B, b_lbound, b_ubound)
	ret, lbound, ubound = banded_dgemt(ret, lbound, ubound, A, a_lbound, a_ubound)
	return ret, lbound, ubound
