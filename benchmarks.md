# Benchmarks

## Adding Benchmarks

To add a new benchmark, one must create a subfolder under [`npbench/benchmarks`](npbench/benchmarks).
This folder must include at least two files.
The first must be called `<benchmark-name>.py` and should contain any
data initialization methods needed for running the benchmark.
The second file is the reference Python/NumPy implementation and must be named
`<benchmark-name>_numpy.py`.
Implementations for other frameworks can also be added.
In general, those implementations should be placed in files named
`<benchmark-name>_<framework-name>.py`.
For example, the CuPy implementation for a benchmark called `mybench` should be
placed in a file with the name `mybench_cupy.py`.
There are exceptions to the above rule, as is the case for Numba.
We expect up to 6 different implementations for Numba named
`<benchmark-name>_numba_<impl>.py`, where `<impl>` is one of
`o, op, opr, n, np, npr`.
`o` indicates that Numba will execute the benchmark in `object` mode, while `n`
corresponds to the `nopython` mode.
`p` indicates that the `parallel` attribute of the `@numba.jit` decorator is
set to `True`, while in the versions with `r` we substitute the Python `range`
iterator in potentially parallel for-loops with `numba.prange`.

## Enabling Benchmarks

After adding the above files, one must enable a new benchmark by creating a
`<benchmark-name>.json` file in the [`bench_info`](bench_info) folder.
The file must be writte in valid JSON and contain the following information:
- name: Name/Description of the benchmark.
- short_name: A short string that is used as the benchmark ID in the result database and heatmaps.
- relative_path: The path to the folder containing the benchmark files relative to `npbench/benchmarks`.
- module_name: The module/filename-prefix of the benchmark files.
- func_name: The name of the function implementing the benchmark.
- kind: Typically `microbench` or `microapp` (optional).
- domain: Scientific domain that the benchmark belongs to, e.g., `Physics`, `Solver`, etc (optional).
- dwarf: The Berkley dwarf represented by the benchmark (optional).
- parameters: The parameters of the benchmark:
  - S: Parameters for small-size runs (~10ms)
  - M: Parameters for medium-size runs (~100ms)
  - L: Parameters for large-size runs (~1s)
  - paper: Parameters used in the NPBench ICS'21 paper
- init: Initialization information:
  - func_name: Name of the initialization function (should exist in `<bechmark-name>.py`).
  - input_args: Input arguments of the initialization function.
  - output_args: Output arguments of the initilization function.
- input_args: Input arguments of the benchmark.
- array_args: The input arguments of the benchmark that are (NumPy) arrays.
- output_args: The input arguments of the benchmark that are also output, i.e., they are written by the benchmark. Use for validation purposes.
