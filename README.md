# NPBench

## Quickstart

To install NPBench, simply execute:
```
python -m pip install -r requirements.txt
python -m pip install .
```
You can then run benchmarks with:
```
python run_benchmark.py -b <benchmark> -f <framework>
```
The available benchmarks are listed in the `bench_info` folder. The available frameworks are listed in the `framework_info` folder. For example, to run ADI with NumPy, execute the following:
```
python run_benchmark.py -b adi -f numpy
```
You can run all the available benchmarks with a specific framework using:
```
python run_framework.py -f <framework>
```
Please note that the NPBench setup only installs NumPy. To run benchmarks with other frameworks, you have to install them separately.  

After running some benchmarks with different frameworks, you can generate plots of the speedups and line-count differences against NumPy:
```
python plot_results.py
python plot_lines.py
```
We provide a quickstart script that runs a few benchmarks with NumPy and Numba and then plots the results:
```
python -m pip install numba
python quickstart.py
python plot_results.py
```

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
