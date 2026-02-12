# Copyright 2025 ETH Zurich and the NPBench authors. All rights reserved.
import pkg_resources

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict, Union, Literal

tl_float: type = None

class TritonFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution("triton").version

    def imports(self) -> Dict[str, Any]:
        return {"torch": __import__("torch")}

    def copy_func(self) -> Callable:
        import torch
        torch.set_default_device('cuda')
        def inner(arr):
            copy = torch.from_numpy(arr).to('cuda')
            return copy
        return inner

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return f"__npb_result = __npb_impl({self.arg_str(bench, impl)}); torch.cuda.synchronize()"

    def set_datatype(self, datatype: Union[Literal["float32"], Literal["float64"]]):
        super().set_datatype(datatype)
        # We might get None here if no datatype is specified. This is sad since we cannot know the exact datatype here
        # and we are relying on the fact that frameworks have their default datatypes set to float32.
        global tl_float
        from triton.language import float32, float64
        tl_float = float64 if datatype == 'float64' else float32


