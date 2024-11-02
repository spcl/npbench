# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


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

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        # param_str = self.param_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        sync_str = """
if isinstance(__npb_result, jax.Array):
    __npb_result.block_until_ready()
elif isinstance(__npb_result, tuple):
    for item in __npb_result:
        if isinstance(item, jax.Array):
            item.block_until_ready()
"""
        return main_exec_str + sync_str
