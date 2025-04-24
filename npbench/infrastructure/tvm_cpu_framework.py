import copy
import numpy
import pkg_resources
from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict
import logging
import sys
import importlib




class TVMCPUFramework(Framework):
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
        from tvm import auto_scheduler
        import tvm
        from tvm import te
        return {'tvm': tvm, 'te': te, 'auto_scheduler': auto_scheduler}

    def copy_func(self) -> Callable:
        """ Returns a method to copy the benchmark arguments to TVM-compatible tensors. """
        import tvm

        def to_tvm_tensor(x):
            if isinstance(x, numpy.ndarray):
                return tvm.nd.array(x)
            else:
                raise TypeError("Input should be a NumPy array.")

        return to_tvm_tensor

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup string to use before calling the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: The corresponding setup-string.
        """


        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args)

        return ""

    def exec_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the execution string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: Execution command string.
        """

        arg_str = self.arg_str(bench, impl)
        main_exec_str = f"__npb_result = __npb_impl({arg_str})"
        return main_exec_str

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
    def autotune(func, name, args, target):
        import tvm
        from tvm import auto_scheduler

        task = auto_scheduler.SearchTask(func=func, args=args, target=target)

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2000,
            measure_callbacks=[auto_scheduler.RecordToFile(f"{name}.json")],
            verbose=2,
        )
        # Run the search
        task.tune(tuning_options=tune_option, search_policy=auto_scheduler.SketchPolicy(task))

        sch, args = task.apply_best(f"{name}.json")

        with tvm.target.Target(target):
            _kernel = tvm.build(sch, args)

        return _kernel