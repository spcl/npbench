import os
import pathlib
import tempfile
from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Any, Callable, Dict, Sequence, Tuple

class NumbaDpexFramework(Framework):
    """ A class for Numba DPEX framework, ensuring data compatibility for SYCL offload. """

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
        import dpnp as np  # Ensuring data is handled in DPNP format
        return {'dpex': dpex, 'dpnp': np}

    def copy_func(self) -> Callable:
        """ Returns a copy function compatible with Numba DPEX, using dpnp arrays. """
        imports = self.imports()
        dpnp = imports['dpnp']
        return dpnp.array  # Ensures arrays are compatible with DPEX

    def copy_back_func(self) -> Callable:
        """ Returns the function to copy data back to a NumPy array if necessary. """
        imports = self.imports()
        dpnp = imports['dpnp']
        return dpnp.asnumpy

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup string for data preparation, copying arrays as needed. """
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return f"{arg_str} = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the execution-string for Numba DPEX, preparing the benchmark call. """
        arg_str = self.arg_str(bench, impl)
        main_exec_str = f"__npb_result = __npb_impl({arg_str})"
        # No explicit synchronization required; Numba DPEX executes synchronously
        return main_exec_str

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        module_pypath = f"npbench.benchmarks.{bench.info['relative_path'].replace('/', '.')}.{bench.info['module_name']}"
        postfix = self.info.get("postfix", self.fname)
        module_str = f"{module_pypath}_{postfix}"
        func_str = bench.info["func_name"]

        ldict = {}
        try:
            exec(f"from {module_str} import {func_str} as impl", ldict)
        except Exception as e:
            print(f"Failed to load the {self.info['full_name']} {func_str} implementation.")
            raise e

        return [(ldict['impl'], 'default')]
