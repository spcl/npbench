# NPBench

## Quickstart

To install NPBench, simply execute:
```
./setup.py
```
You can then run benchmarks with:
```
python run_benchmark.py -b <benchmark> -f <framework>
```
The available benchmarks are listed in the `bench_info` folder. The available frameworks are listed in the `framework_info` folder. For example, to run ADI with NumPy, execute the following:
```
python run_benchmark.py -b adi -f numpy
```
Please note that the NPBench setup only installs NumPy. To run benchmarks with other frameworks, you have to install them separately.

## Supported Frameworks

Currently, the following frameworks are supported (in alphabetical order):
- CuPy
- DaCe (CPU and GPU)
- Numba
- NumPy
- Pythran

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

### DaCe

TODO

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

## Adding Benchmarks

TODO

## Adding Frameworks

TODO
