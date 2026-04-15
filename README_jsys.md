## Installation

We recommend a conda installation and provide a `environment.yml` file for that.
You can create the conda environment with:
```bash
$ conda env create -f environment.yml  # environment.yml contains all dependencies
$ conda activate npb  # Activate the environment
$ python -m pip install . # Install npbench package
```

This will install all dependencies for running on CPU with NumPy, Numba, and DPNP.
If you have a suitable GPU, you can also (optionally) install CuPy, following the instructions here: [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).


## Verifying the installation

To check that the installation was successful, you can run the following command:
```bash
$ python run_framework_jsys.py --framework numpy --preset S
```
The command will executed selected benchmarks presented in the paper with NumPy and the smallest input size (preset S).
If the command runs successfully, you should see output similar to the following:
```
***** Testing NumPy with adi on the S dataset *****
NumPy - default - first/validation: 11ms
NumPy - default - median: 11ms
***** Testing NumPy with jacobi_1d on the S dataset *****
NumPy - default - first/validation: 6ms
NumPy - default - median: 6ms
***** Testing NumPy with jacobi_2d on the S dataset *****
NumPy - default - first/validation: 5ms
NumPy - default - median: 5ms
...
```

## Minimal functional reproduction

After successfully checking basic functionality, you can run a minimal reproduction of the functionality of the DPNP benchmark implementations with the following command:
```bash
$ python run_framework_jsys.py --framework dpnp_cpu --preset S
```
This will run the DPNP CPU benchmarks with the smallest input size (preset S).
Please note that a few benchmarks may not run successfully and this is expected.
Then you can visualize the results, similarly to Figure 1 in the paper, with the following command:
```bash
$ python plot_results.py --preset S
```
This will generate a heatmap of the results and save it as `heatmap.pdf` in the current directory.
Plase note that at those small input sizes, the results may be noisy and DPNP may not perform well, but this is expected.

## Reproducing the paper results

Assuming the availability of the required hardware, to reproduce the results of the paper, you can run the following command:
```bash
$ python run_framework_jsys.py --framework <framework> --preset paper
```
This will run all benchmarks with the input sizes used in the paper (preset paper).
Here, `<framework>` should be replaced with the name of the framework you want to run (`numpy`, `dpnp_cpu`, `numba`).
If you have a suitable GPU and sofware stack, you can also run `dpnp_gpu` and `cupy`.
Please note that certain combinations of benchmarks and frameworks may not run successfully, and this is expected.
After running the benchmarks, you can visualize the results with the following command:
```bash
$ python plot_results.py --preset paper
```
This will generate a heatmap of the results and save it as `heatmap.pdf` in the current directory.
Please note that before visualization, you need to run the benchmarks with NumPy, as it is used as a reference for the speedups shown in the heatmap.
If NumPy results are not available, the heatmap will not be generated, and this is expected.

## Troubleshooting

### Timeouts
Some benchmarks may not complete within a reasonable time, and this is expected.
By default, benchmarks that do not complete within 200 seconds are considered to have failed and are not included in the results.
This threshold can be adjusted with the `--timeout` flag when running the benchmarks, as described below:
```bash
$ python run_framework_jsys.py --framework <framework> --preset paper --timeout <timeout_in_seconds>
```
If you want to include in your evaluation benchmarks that do not complete within the timeout threshold, we suggest running them individually with the `run_benchmark.py` script:
```bash
$ python run_benchmark.py -b <benchmark> -f <framework> -p paper -t <timeout_in_seconds>
```
This way, you can avoid having to re-run all benchmarks if a few of them do not complete within the timeout threshold.

