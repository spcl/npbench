import numpy
import pkg_resources
from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict
import logging
import sys
import tvm
import tvm.testing
from tvm import autotvm
import importlib


class TVMFramework(Framework):
    """ A class for reading and processing framework information specific to TVM. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """
        super().__init__(fname)

    def version(self) -> str:
        """ Return the TVM framework version. """
        return [p.version for p in pkg_resources.working_set if p.project_name.startswith("tvm")][0]

    def imports(self) -> Dict[str, Any]:
        """ Import TVM-specific modules. """
        import tvm
        from tvm import te
        import tvm.testing
        from tvm import autotvm
        import torch
        return {'tvm': tvm, 'te': te, 'autotvm': autotvm, 'torch': torch}

    def copy_func(self) -> Callable:
        """ Returns a method to copy the benchmark arguments to TVM-compatible tensors. """
        import torch
        import tvm

        def to_tvm_tensor(x):
            if isinstance(x, numpy.ndarray):
                # Convert NumPy array to Torch tensor on GPU
                return tvm.nd.array(x, tvm.cuda(0))
            elif isinstance(x, torch.Tensor):
                raise TypeError("Input should be either a NumPy array.")
            else:
                raise TypeError("Input should be either a NumPy array.")

        return to_tvm_tensor

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

    @staticmethod
    def autotune(kernel_name, module_name, args_tuple, target):
        module = importlib.import_module(module_name)
        func = getattr(module, kernel_name)

        task1 = autotvm.task.create(kernel_name, args=args_tuple, target=target)

        logging.getLogger("autotvm").setLevel(logging.DEBUG)
        logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=1, do_fork=False),
            runner=autotvm.LocalRunner(number=1, repeat=1)
        )
        tuner1 = autotvm.tuner.RandomTuner(task1)
        tuner1.tune(
            n_trial=100,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(f"{kernel_name}.log")],
        )

        with autotvm.apply_history_best(f"{kernel_name}.log"):
            with target:
                s1, arg_bufs1 = func(*args_tuple)
                _kernel = tvm.build(s1, arg_bufs1)
                return _kernel