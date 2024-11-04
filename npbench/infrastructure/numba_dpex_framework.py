import os
import pathlib
import tempfile
from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Any, Callable, Dict, Sequence, Tuple

class NumbaDpexFramework(Framework):
    """ A class for reading and processing framework information specific to Numba DPEX. """

    def __init__(self, fname: str):
        """ Initializes the Numba DPEX framework with the provided name. """
        super().__init__(fname)

    def version(self) -> str:
        """ Returns the Numba DPEX version. """
        try:
            dpex = __import__('numba_dpex')
            return dpex.__version__
        except ImportError:
            return "Numba DPEX not installed"

    def imports(self) -> Dict[str, Any]:
        """ Returns the required imports for Numba DPEX. """
        import numba_dpex as dpex
        return {'dpex': dpex}

    def copy_func(self) -> Callable:
        """ Returns the copy method for Numba DPEX arrays. 
        Numba DPEX often works with NumPy arrays or similar constructs.
        """
        import numpy as np
        return np.array  # Copy to a NumPy array; `numba_dpex` can operate on NumPy arrays.

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup string, typically for copying data to SYCL device. """
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the execution-string for Numba DPEX, preparing the benchmark call. """
        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        # Numba DPEX operates synchronously by default, no explicit synchronization needed.
        return main_exec_str

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the implementations specific to Numba DPEX for a particular benchmark. """
        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        ldict = dict()
        try:
            exec("from {m} import {f} as impl".format(m=module_str, f=func_str), ldict)
        except Exception as e:
            print(f"Failed to load the {self.info['full_name']} {func_str} implementation.")
            raise e

        return [(ldict['impl'], 'default')]

