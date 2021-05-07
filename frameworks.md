# Frameworks

# Adding Frameworks

To add a new framework, one must add a file `<framework-name>.json` in the `framework_info` folder.
The file must be written in valid JSON and must contain the following information:
- simple_name: Should be the same as `<framework-name>`.
- full_name: The full name of the framework. Used for reporting purposes.
- prefix: A short string used as a prefix on the names of, for example, data copies that are created by the NPBench runtime.
- postfix: The postix used in the filenames of the framework's benchmark implementations.
- class: The `Framework` (sub)class that is used by the NPBench runtime. See below for more information.
- arch: The architecture that the framework is running on, e.g. `cpu`, `gpu`, etc.

# Framework Class

NPBench uses a `Framework` (sub)class to initialize a framework's information and to generate all the structures needed to run the benchmarks with the framework.
The base `Framework` class (found in `npbench/infrastructure/framework.py`) contains the following methods:
- version: Returns the framework's version. Used for storing the results in the database.
- imports: Create a dictionary of modules and methods needed to run a benchmark with the framework.
- copy_func: Returns the method that should be used for making copying of the benchmark's input (array) arguments. This can be, for example, `numpy.copy` or `cupy.asarray` (for copying the data directly to the GPU).
- impl_files: Returns a list of the framework's implementation files for the input benchmark. Each element in the list is a tuple of the implementation filename and a description (e.g. `default` or `nopython-parallel`).
- implementations: Returns a list of the framework's implementations for the input benchmark. Each element in the list is a tuple of the implementation method and a description (as above).
- args: Returns a list with the names of the input arguments for running the input implementation of the input benchmark.
- out_args: Returns a list with the input arguments for running the input implementation of the input benchmark **and** have to be copied(for example, because they may be modified during benchmark execution).
- arg_str: Returns the argument-string needed to call the input implementation of the input benchmark.
- out_arg_str: Returns the argument-string with the input arguments that must be copied.
- setup_str: Returns the setup-string of the code that should be executed for, e.g., copying data, before executing the benchmark implementation.
- exec_str: Returns the execution-string for executing the benchmark implementation.

It is very likely that just by setting the `class` field in the framework's JSON file to `Framework`, everything will work. If your framework needs some kind of special handling, you must create your own class that inherits from `Framework` and specializes any of the above methods as needed.
Some typical scenarios where you want to do so are:
- Your framework's version cannot be found with the `pkg_resources` module (see `npbench/infrastructure/cupy_framework.py`).
- Your framework needs some special method during setup/execution that you want to import only once.
- You want to use a special copy function. For example, we use `cupy.asarray` instead of `numpy.copy` when executing the benchmarks on the GPU, so that we avoid measuring the copy of data from host to device and back.
- Your framework has multiple implementations for each benchmark (see `npbench/infrastructure/numba_framework.py`)
- Your framework needs some kind of synchronization before and after executing a benchmark. Therefore, you need to specialize the setup and execution string (see `npbench/infrastructure/cupy_framework.py`).

If you create a `Framework` subclass, remember to also to add a line in `npbench/infrastructure/__init__.py`: `from <framework-subclass-filename> import *`.
