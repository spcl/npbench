import os
import pathlib
import tempfile
import dpnp as np
from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Any, Callable, Dict, Sequence, Tuple

class DpcppFramework(Framework):
    """ A class for reading and processing framework information specific to DPC++/Dpnp. """

    def __init__(self, fname: str):
        """ Initializes the DPC++ framework with the provided name. """
        super().__init__(fname)

    def version(self) -> str:
        """ Returns the Dpnp version. """
        return np.__version__

    def imports(self) -> Dict[str, Any]:
        """ Returns the required imports for Dpnp. """
        import dpnp
        return {'dpnp': dpnp}

    def copy_func(self) -> Callable:
        """ Returns the copy method for Dpnp, typically dpnp.asarray. """
        import dpnp
        return dpnp.asarray

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup-string, typically for copying data to the device. """
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string for Dpnp. """
        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        # Synchronization might not be necessary, but if Dpnp supports it, add it here.
        # sync_str = "dpnp.synchronize()"  # Uncomment if dpnp has synchronization
        return main_exec_str  # + "; " + sync_str if synchronization is needed

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the implementations specific to Dpnp for a particular benchmark. """
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
            print("Failed to load the {r} {f} implementation.".format(r=self.info["full_name"], f=func_str))
            raise e

        return [(ldict['impl'], 'default')]
