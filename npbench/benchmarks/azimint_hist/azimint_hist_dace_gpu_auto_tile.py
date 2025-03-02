# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

# BSD 2-Clause License

# Copyright (c) 2017, Numba
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import dace as dc

N, bins, npt = (dc.symbol(s, dtype=dc.int64) for s in ('N', 'bins', 'npt'))


@dc.program
def get_bin_edges(a: dc.float64[N], bin_edges: dc.float64[bins + 1]):
    a_min = np.amin(a)
    a_max = np.amax(a)
    delta = (a_max - a_min) / bins
    for i in dc.map[0:bins]:
        bin_edges[i] = a_min + i * delta

    bin_edges[bins] = a_max  # Avoid roundoff error on last point


@dc.program
def compute_bin(x: dc.float64, bin_edges: dc.float64[bins + 1]):
    # assuming uniform bins for now
    a_min = bin_edges[0]
    a_max = bin_edges[bins]
    return dc.int64(bins * (x - a_min) / (a_max - a_min))


@dc.program
def histogram(a: dc.float64[N], bin_edges: dc.float64[bins + 1]):
    hist = np.ndarray((bins, ), dtype=np.int64)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += 1

    return hist


@dc.program
def histogram_weights(a: dc.float64[N], bin_edges: dc.float64[bins + 1],
                      weights: dc.float64[N]):
    hist = np.ndarray((bins, ), dtype=weights.dtype)
    hist[:] = 0
    get_bin_edges(a, bin_edges)

    for i in dc.map[0:N]:
        bin = min(compute_bin(a[i], bin_edges), bins - 1)
        hist[bin] += weights[i]

    return hist


@dc.program
def _azimint_hist(data: dc.float64[N], radius: dc.float64[N]):
    # histu = np.histogram(radius, npt)[0]
    bin_edges_u = np.ndarray((npt + 1, ), dtype=np.float64)
    histu = histogram(radius, bin_edges_u)
    # histw = np.histogram(radius, npt, weights=data)[0]
    bin_edges_w = np.ndarray((npt + 1, ), dtype=np.float64)
    histw = histogram_weights(radius, bin_edges_w, data)
    r = histw / histu
    return r

_best_config = None

def autotuner(data, radius):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _azimint_hist.to_sdfg(),
        {"data": data, "radius": radius},
        dims=get_max_ndim([data, radius])
    )

def azimint_hist(data, radius):
    global _best_config
    r = _best_config(data, radius)
    return r
