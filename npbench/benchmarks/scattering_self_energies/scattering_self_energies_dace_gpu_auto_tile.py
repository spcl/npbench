# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import numpy as np
import dace as dc

NA, NB, Nkz, NE, Nqz, Nw, Norb, N3D = (dc.symbol(s, dc.int64)
                                       for s in ('NA', 'NB', 'Nkz', 'NE',
                                                 'Nqz', 'Nw', 'Norb', 'N3D'))


@dc.program
def _scattering_self_energies(neigh_idx: dc.int32[NA, NB],
                             dH: dc.complex128[NA, NB, N3D, Norb, Norb],
                             G: dc.complex128[Nkz, NE, NA, Norb, Norb],
                             D: dc.complex128[Nqz, Nw, NA, NB, N3D, N3D],
                             Sigma: dc.complex128[Nkz, NE, NA, Norb, Norb]):

    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w,
                                                neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD

_best_config = None

def autotuner(neigh_idx, dH, G, D, Sigma):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _scattering_self_energies.to_sdfg(),
        {"neigh_idx": neigh_idx, "dH": dH, "G": G, "D": D, "Sigma": Sigma},
        dims=get_max_ndim([neigh_idx, dH, G, D, Sigma])
    )

def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
    global _best_config
    _best_config(neigh_idx, dH, G, D, Sigma)
    return Sigma
