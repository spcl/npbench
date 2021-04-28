# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True, fastmath=True)
def scattering_self_energies(neigh_idx, dH, G, D, Sigma):

    for k in nb.prange(G.shape[0]):
        for E in nb.prange(G.shape[1]):
            for q in nb.prange(D.shape[0]):
                for w in nb.prange(D.shape[1]):
                    for i in nb.prange(D.shape[-2]):
                        for j in nb.prange(D.shape[-1]):
                            for a in nb.prange(neigh_idx.shape[0]):
                                for b in nb.prange(neigh_idx.shape[1]):
                                    if E - w >= 0:
                                        dHG = G[k, E - w,
                                                neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
