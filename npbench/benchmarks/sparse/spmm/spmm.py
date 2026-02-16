# Copyright 2023 University Politehnica of Bucharest and the NPBench authors. All rights reserved.

import numpy as np

def initialize(NI, NJ, NK, nnz_A, nnz_B, datatype=np.float64):
	import scipy.sparse as sp
	rng = np.random.default_rng(42)
	alpha = datatype(0.8)
	beta = datatype(0.3)
	np.random.seed(42)
	C = np.fromfunction(lambda i, j: np.random.rand(i.shape[0], j.shape[0]), (NI, NJ), dtype=datatype)
	A = sp.random(NI, NK, density=nnz_A / (NI * NK), format='csr', dtype=datatype, random_state=rng)
	B = sp.random(NK, NJ, density=nnz_B / (NK * NJ), format='csr', dtype=datatype, random_state=rng)
	return alpha, beta, C, A, B
