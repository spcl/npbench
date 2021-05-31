<img src="npbench.svg" alt="npbench-logo" width="100"/>
<h1>NPBench</h1>

## Quickstart

To install NPBench, simply execute:
```
python -m pip install -r requirements.txt
python -m pip install .
```
You can then run a subset of the benchmarks with NumPy, Numba, and DaCe and plot
the speedup of DaCe and Numba against NumPy:
```
python -m pip install numba
python -m pip install dace
python quickstart.py
python plot_results.py
```

## Supported Frameworks

Currently, the following frameworks are supported (in alphabetical order):
- CuPy
- Numba
- NumPy
- Pythran

Support will also be added shortly for:
- DaCe
- Legate

Please note that the NPBench setup only installs NumPy.
To run benchmarks with other frameworks, you have to install them separately.
Below, we provide some tips about installing each of the above frameworks:

### CuPy

If you already have CUDA installed, then you can install CuPy with pip:
```
python -m pip install cupy-cuda<version>
```
For example, if you have CUDA 11.1, then you should install CuPy with:
```
python -m pip install cupy-cuda111
```
For more installation options, consult the CuPy [installation guide](https://docs.cupy.dev/en/stable/install.html#install-cupy).

### Numba

Numba can be installed with pip:
```
python -m pip install numba
```
If you use Anaconda on an Intel-based machine, then you can install an optimized version of Numba that uses Intel SVML:
```
conda install -c numba icc_rt
```
For more installation options, please consult the Numba [installation guide](https://numba.readthedocs.io/en/stable/user/installing.html).

### Pythran

Pythran can be install with pip and Anaconda. For detailed installation options, please consult the Pythran [installation guide](https://pythran.readthedocs.io/en/latest/).


## Running benchmarks

To run individual bencharks, you can use the `run_benchmark` script:
```
python run_benchmark.py -b <benchmark> -f <framework>
```
The available benchmarks are listed in the `bench_info` folder.
The supported frameworks are listed in the `framework_info` folder.
Please use the corresponding JSON filenames.
For example, to run `adi` with NumPy, execute the following:
```
python run_benchmark.py -b adi -f numpy
```
You can run all the available benchmarks with a specific framework using the `run_framework` script:
```
python run_framework.py -f <framework>
```

### Presets

Each benchmark has four different presets; `S`, `M`, `L`, and `paper`.
The `S`, `M`, and `L` presets have been selected so that NumPy finishes execution
in about 10, 100, and 1000ms respectively in a machine with two 16-core Intel Xeon
Gold 6130 processors.
Exception to that are `atax`, `bicg`, `mlp`, `mvt`, and `trisolv`, which have been
tuned for 5, 20 and 100ms approximately due to very high memory requirements.
The `paper` preset is the problem sizes used in the NPBench [paper](http://spcl.inf.ethz.ch/Publications/index.php?pub=412).
By default, the provided python scripts execute the benchmarks using the `S` preset.
You can select a different preset with the optional `-p` flag:
```
python run_benchmark.py -b gemm -f numpy -p L
```

### Visualization

After running some benchmarks with different frameworks, you can generate plots
of the speedups and line-count differences (experimental) against NumPy:
```
python plot_results.py
python plot_lines.py
```

## Customization

It is possible to use the NPBench infrastructure with your own benchmarks and frameworks.
For more information on this functionality please read the documentation for [benchmarks](benchmarks.md) and [frameworks](frameworks.md).

## Acknowledgements

NPBench is a collection of scientific Python/NumPy codes from various domains that we adapted from the following sources:
- Azimuthal Integration from [pyFAI](https://github.com/silx-kit/pyFAI)
- Navier-Stokes from  [CFD Python](https://github.com/barbagroup/CFDPython)
- Cython [tutorial](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html) for NumPy users
- Quantum Transport simulation from [OMEN](https://nano-tcad.ee.ethz.ch/research/computational-nanoelectronics.html)
- CRC-16-CCITT algorithm from [oysstu](https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6)
- Numba [tutorial](https://numba.readthedocs.io/en/stable/user/5minguide.html)
- Mandelbrot codes [From Python to Numpy](https://github.com/rougier/from-python-to-numpy)
- N-Body simulation from [nbody-python](https://github.com/pmocz/nbody-python)
- [PolyBench/C](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/)
- Pythran [benchmarks](https://github.com/serge-sans-paille/numpy-benchmarks/)
- [Stockham-FFT](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-287731)
- Weather stencils from [gt4py](https://github.com/GridTools/gt4py)




