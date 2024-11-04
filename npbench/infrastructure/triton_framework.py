import pkg_resources
from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class TritonFramework(Framework):
    """A class for reading and processing framework information specific to Triton."""

    def __init__(self, fname: str):
        """Reads framework information.
        :param fname: The framework name.
        """
        super().__init__(fname)

    def version(self) -> str:
        """Return the Triton framework version."""
        return [
            p.version for p in pkg_resources.working_set if p.project_name == "triton"
        ][0]

    def imports(self) -> Dict[str, Any]:
        import triton
        import triton.language as tl

        return {"triton": triton, "tl": tl}

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """Generates the setup string to prepare for calling the benchmark.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: The corresponding setup-string.
        """
        sync_str = "torch.cuda.synchronize()"
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            # Triton uses PyTorch tensors for GPU data; no need for an explicit copy function
            copy_args = [
                f"{a}.clone().to(device='cuda')" for a in bench.info["array_args"]
            ]
            return f"{arg_str} = {', '.join(copy_args)}; {sync_str}"
        return sync_str

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """Generates the execution-string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """
        arg_str = self.arg_str(bench, impl)
        main_exec_str = f"__npb_result = __npb_impl({arg_str})"
        sync_str = "torch.cuda.synchronize()"
        return f"{main_exec_str}; {sync_str}"
