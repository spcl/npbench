# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pkg_resources

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class APPyFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return 0.1

    # def copy_func(self) -> Callable:
    #     """ Returns the copy-method that should be used 
    #     for copying the benchmark arguments. """
    #     import cupy
    #     return cupy.asarray

    def copy_func(self) -> Callable:
        import torch
        torch.set_default_device('cuda')
        def inner(arr):
            copy = torch.from_numpy(arr).to('cuda')
            return copy
        return inner

    def imports(self) -> Dict[str, Any]:
        import torch
        import appy
        return {'torch': torch}

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        # param_str = self.param_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        sync_str = "torch.cuda.synchronize()"
        return main_exec_str + "; " + sync_str
