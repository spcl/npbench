# Copyright 2023 University Politehnica of Bucharest and the NPBench authors. All rights reserved.

import numpy as np

# Generate required elements for a matrix with at least one non-zero term
# in the determinant
def required_elems(n):
	rows = np.random.choice(n, n, replace=False)
	cols = np.random.choice(n, n, replace=False)
	vals = np.array(np.random.rand(n) * 10 - 5, dtype=np.float64)
	return to_symetric(rows, cols, vals)

# Generate more elements to make sure there are nnz or nnz + 1
def ensure_nnz(n, nnz, rows, cols, vals):
	nnz -= len(rows)
	if nnz <= 0:
		return
	coords = set(zip(rows, cols))
	# Generate at least as many random values as they are required
	new_vals = np.array(np.random.rand(nnz) * 10 - 5, dtype=np.float64)
	i = 0
	while nnz > 0:
		x, y = np.random.choice(n, 2)
		stop = y
		# The following loop changes x and y untill they are a new pair of coordinates
		while (x, y) in coords:
			y += 1
			if y == n:
				y = 0
			if y == stop:
				x += 1
				if x == n:
					x = 0
		generated_pair = (x, y)
		coords.add(generated_pair)
		rows.append(generated_pair[0])
		cols.append(generated_pair[1])
		vals.append(new_vals[i])
		nnz -= 1
		# Adds the symetric of the newly added element in the set
		if not generated_pair[0] == generated_pair[1]:
			coords.add((generated_pair[1], generated_pair[0]))
			rows.append(generated_pair[1])
			cols.append(generated_pair[0])
			vals.append(new_vals[i])
			nnz -= 1
		i += 1

# Adds the symetrics of the existent elements in the matrix
def to_symetric(rows, cols, vals):
	new_cols, new_rows, new_vals = [], [], []
	for i in range(0, vals.size):
		new_rows.append(rows[i])
		new_cols.append(cols[i])
		new_vals.append(vals[i])
		if not rows[i] == cols[i]:
			new_rows.append(cols[i])
			new_cols.append(rows[i])
			new_vals.append(vals[i])
	return new_rows, new_cols, new_vals

def initialize(n : int, nnz : int, dtype=np.float64):
	np.random.seed(42)
	rows, cols, vals = required_elems(n)
	ensure_nnz(n, nnz, rows, cols, vals)
	from scipy.sparse import coo_matrix
	A = coo_matrix((vals, (rows, cols)), shape=(n, n)).asformat('csr')
	b = A @ np.random.rand(n)
	# Generate starting solution
	x = np.random.rand(n)
	return A, x, b
