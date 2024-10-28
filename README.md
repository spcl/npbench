<img src="npbench.svg" alt="npbench-logo" width="100"/>
<h1>NPBench</h1>

<br>
<br>


- [Installation](#installation)
  - [...on a VM in the LRZ Compute Cloud (with only VCPUs or with Nvidia v100 GPUs)](#on-a-vm-in-the-lrz-compute-cloud-with-only-vcpus-or-with-nvidia-v100-gpus)
  - [...on SuperMUC-NG Phase 1](#on-supermuc-ng-phase-1)
  - [...on SuperMUC-NG Phase 2](#on-supermuc-ng-phase-2)
  - [Conda env without licensing issues](#conda-env-without-licensing-issues)
- [Supported Frameworks](#supported-frameworks)
  - [CuPy](#cupy)
- [Running benchmarks](#running-benchmarks)
  - [Presets](#presets)
  - [Visualization](#visualization)
- [Customization](#customization)
- [Publication](#publication)
- [Acknowledgements](#acknowledgements)




## Installation 

First of all, put a copy of this directory where you want to run the benchmarkings; you can use scp, git or whatever other way you prefer.

To install this branch NPBench, including dpnp/numba-dpex, we can use conda for most of the packages and pip for dpnp itself.

NOTE: You could, in line of principle, install DPNP with Pip, but if you do so, it will not see the GPUs!

```bash
$ python -m pip --proxy=http://localhost:1234 install dpnp      # where "localhost:1234" is the value of the env var "HTTP_PROXY"
```

NOTE: On SuperMUC-NG Phase 1/2, you need to have internet connection (to allow conda/pip to download packages) => SSH Remote Forward

<br>
<br>
<br>






### ...on a VM in the LRZ Compute Cloud (with only VCPUs or with Nvidia v100 GPUs)

On a CC instance, we have internet connection => no SSH Remote Forward necessary.<br>
However, there are no modules to be loaded, so we need to install oneAPI with get:


```bash
$ conda env create -f environment.yml 
$ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100_offline.sh
```


<br>
<br>
<br>










### ...on SuperMUC-NG Phase 1

On SuperMUC-NG Phase 1, with Spack 22.2.1:

```bash
# suppose you have set up internet connection correctly
$ env | grep -iE "HTTP_|HTTPS_"
    HTTP_PROXY=localhost:1234
    https_proxy=localhost:1234
    http_proxy=localhost:1234
    HTTPS_PROXY=localhost:1234
$ module list spack
    Currently Loaded Matching Modulefiles:
        1) spack/22.2.1  

    Key:
    default-version  
$ conda env create -f environment.yml                             # environment.yml contains all the right dependencies
```

<br>
<br>
<br>







### ...on SuperMUC-NG Phase 2


On SuperMUC-NG Phase 2, in addition to what shown for Phase 1, you either 

- swap from Spack 24.1.0 to Spack 22.2.1 (currently not available however)
- install with pip the package pygount

Here we show the second way:

```bash
# suppose you have set up internet connection correctly
$ env | grep -iE "HTTP_|HTTPS_"
    HTTP_PROXY=localhost:1234
    https_proxy=localhost:1234
    http_proxy=localhost:1234
    HTTPS_PROXY=localhost:1234
$ module list spack
    Currently Loaded Modulefiles:
    1) admin/2.0   2) tempdir/1.0   3) lrz/1.0   4) mpi_settings/1.0   5) user_spack/24.1.0   6) lrztools/2.0  

    Key:
    default-version  sticky  
$ conda env create -f environment.yml                           # environment.yml contains all the right dependencies
$ python -m pip --proxy=http://localhost:1234 install pygrount  # where "localhost:1234" is the value of the env var "HTTP_PROXY"
```



<br>
<br>
<br>









### Conda env without licensing issues

Creating the conda environment without licensing issues:

1. download & install miniconda (or something similar)
2. conda init as normal
3. edit .condarc
4. modify the channels an put them in this order:

```yaml
channel:
  - https://software.repos.intel.com/python/conda
  - conda-forge
  - defaults              #  you can either remove or put this "defaults" channel last; the important thing is that conda won't try to use this channel
```

<br>
<br>
<br>
<br>
<br>




















## Supported Frameworks

Currently, the following frameworks are supported (in alphabetical order):
- CuPy
- DaCe
- Dpnp
- Numba
- Numba-dpex
- NumPy
- Pythran

Support will also be added shortly for:
- Legate

Please note that the NPBench setup installs all frameworks except CuPy.
Below, we provide some tips about installing each of the above frameworks:

<br>
<br>
<br>




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

<br>
<br>
<br>
<br>
<br>












## Running benchmarks

You can then run a subset of the benchmarks with NumPy, Numba, DaCe, Dpnp and Numba-dpex,
and plot the speedup of DaCe and Numba against NumPy:

```bash
$ python main.py
$ python plot_results.py
```

To run individual bencharks, you can use the `$ run_benchmark` script:

```bash
$ python run_benchmark.py -b <benchmark> -f <framework>
```

- the available benchmarks are listed in the `bench_info` folder
- the supported frameworks are listed in the `framework_info` folder
- lease use the corresponding JSON filenames for both of them


For example, to run `adi` with NumPy, execute the following:

```bash
$ python run_benchmark.py -b adi -f numpy
```

You can run all the available benchmarks with a specific framework using the `run_framework` script:

```bash
python run_framework.py -f <framework>
```

<br>
<br>
<br>





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

```bash
python run_benchmark.py -b gemm -f numpy -p L
```

<br>
<br>
<br>




### Visualization

After running some benchmarks with different frameworks, you can generate plots
of the speedups and line-count differences (experimental) against NumPy:


```bash
$ python plot_results.py
$ python plot_lines.py
```

<br>
<br>
<br>
<br>
<br>












## Customization

It is possible to use the NPBench infrastructure with your own benchmarks and frameworks.
For more information on this functionality please read the documentation for [benchmarks](benchmarks.md) and [frameworks](frameworks.md).

## Publication

Please cite NPBench as follows:

```bibtex
@inproceedings{
    npbench,
    author = {Ziogas, Alexandros Nikolaos and Ben-Nun, Tal and Schneider, Timo and Hoefler, Torsten},
    title = {NPBench: A Benchmarking Suite for High-Performance NumPy},
    year = {2021},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3447818.3460360},
    doi = {10.1145/3447818.3460360},
    booktitle = {Proceedings of the ACM International Conference on Supercomputing},
    series = {ICS '21}
}
```

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




