# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import importlib
import pathlib

try:
    import jax.numpy as jnp
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    print("WARNING: JAX is not installed. " 
          "Please install JAX to run benchmarks with the JAX framework.")

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


_impl = {
    'lib-implementation': 'lib'
}

class JaxFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def imports(self) -> Dict[str, Any]:
        return {'jax': jax}

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """
        return jnp.array

    def impl_files(self, bench: Benchmark):
        """ Returns the framework's implementation files for a particular
        benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementation files.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        implementations = []

        # appending the default implementation
        pymod_path = parent_folder.joinpath("..", "..", "npbench", "benchmarks", bench.info["relative_path"],
                                            bench.info["module_name"] + "_" + self.info["postfix"] + ".py")
        
        implementations.append((pymod_path, 'default'))

        for impl_name, impl_postfix in _impl.items():
            pymod_path = parent_folder.joinpath(
                "..", "..", "npbench", "benchmarks", bench.info["relative_path"],
                bench.info["module_name"] + "_" + self.info["postfix"] + "_" + impl_postfix + ".py")
            implementations.append((pymod_path, impl_name))
        
        return implementations
    
    def implementations(self, bench: Benchmark):
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        implementations = []

        # appending the default implementation
        try:
            ldict = dict()
            
            module = importlib.import_module(module_str)
            ldict['impl'] = getattr(module, func_str)
            implementations.append((ldict['impl'], 'default'))
        except Exception as e:
            print("Failed to load the {r} {f} implementation.".format(r=self.info["full_name"], f=func_str))
            raise e

        for impl_name, impl_postfix in _impl.items():
            ldict = dict()
            try:
                module = importlib.import_module("{m}_{p}".format(m=module_str, p=impl_postfix))
                ldict['impl'] = getattr(module, func_str)
                implementations.append((ldict['impl'], impl_name))
            except ImportError:
                continue
            except Exception:
                print("Failed to load the {r} {f} implementation.".format(r=self.info["full_name"], f=impl_name))
                continue
        
        return implementations

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__npb_result = jax.block_until_ready(__npb_impl({a}))".format(a=arg_str)
    
        return main_exec_str
