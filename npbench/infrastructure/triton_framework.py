import numpy
import pkg_resources
from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class TritonFramework(Framework):
    """ A class for reading and processing framework information specific to Triton. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """
        super().__init__(fname)

    def version(self) -> str:
        """ Return the Triton framework version. """
        return [p.version for p in pkg_resources.working_set if p.project_name.startswith("triton")][0]

    def imports(self) -> Dict[str, Any]:
        """ Import Triton-specific modules. """
        import triton
        import triton.language as tl
        import torch
        return {'triton': triton, 'tl': tl, 'torch': torch}

    def copy_func(self) -> Callable:
        """ Returns a method to copy the benchmark arguments to Triton-compatible tensors. """
        import torch

        def to_triton_tensor(x):
            if isinstance(x, numpy.ndarray):
                # Convert NumPy array to Torch tensor on GPU
                return torch.tensor(x, device='cuda')
            elif isinstance(x, torch.Tensor):
                # Move existing Torch tensor to GPU if not already there
                return x.to('cuda')
            else:
                raise TypeError("Input should be either a NumPy array or a Torch tensor.")

        return to_triton_tensor

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup string to use before calling the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: The corresponding setup-string.
        """

        # Synchronize CUDA streams for consistent measurements
        sync_str = "torch.cuda.synchronize()"
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args) + "; " + sync_str
        return sync_str

    def exec_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the execution string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: Execution command string.
        """

        arg_str = self.arg_str(bench, impl)
        main_exec_str = f"__npb_result = __npb_impl({arg_str})"
        sync_str = "torch.cuda.synchronize()"
        return main_exec_str + "; " + sync_str

    def autotune_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the execution string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: Execution command string.
        """

        arg_str = self.arg_str(bench, impl)
        autotune_str = f"__npb_autotune_result =__npb_autotune({arg_str})"
        return autotune_str