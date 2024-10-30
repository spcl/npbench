import os
import pathlib
import tempfile

from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Any, Callable, Dict, Sequence, Tuple

class DpnpFramework(Framework):
    """ A class for DPC++/DPNP framework supporting both CPU and GPU selection. """

    def __init__(self, fname: str):
        """ Initializes the DPC++ framework with the provided name. """
        super().__init__(fname)
        self.device_type = None  # Store the device type based on the framework

    def version(self) -> str:
        """ Returns the Dpnp version. """
        try:
            dpnp = __import__('dpnp')
            return dpnp.__version__
        except ImportError:
            return "DPNP not installed"

    def imports(self) -> Dict[str, Any]:
        """ Returns the required imports for Dpnp and dpctl. """
        import dpnp
        import dpctl
        return {'dpnp': dpnp, 'dpctl': dpctl}

    def select_device(self):
        """ Selects the default SYCL device based on the framework name (CPU or GPU). """
        imports = self.imports()  # Ensure imports are correctly fetched
        dpctl = imports['dpctl']

        # Identify the framework type and select the appropriate device
        if self.fname == "dpnp_cpu":
        # Select CPU device
            cpu_device = dpctl.select_cpu_device()
            selector = f"{cpu_device.backend.name}:{cpu_device.device_type.name}"
            os.environ['ONEAPI_DEVICE_SELECTOR'] = selector
            print(f"Selected CPU device: {cpu_device}")
            print(f"Set ONEAPI_DEVICE_SELECTOR to {selector}")
            return cpu_device
        elif self.fname == "dpnp_gpu":
        # Select GPU device
            gpu_device = dpctl.select_gpu_device()
            selector = f"{gpu_device.backend.name}:{gpu_device.device_type.name}"
            os.environ['ONEAPI_DEVICE_SELECTOR'] = selector
            print(f"Selected GPU device: {gpu_device}")
            print(f"Set ONEAPI_DEVICE_SELECTOR to {selector}")
            return gpu_device
        else:
        # Default device (usually GPU if available)
            default_device = dpctl.select_default_device()
            selector = f"{default_device.backend.name}:{default_device.device_type.name}"
            os.environ['ONEAPI_DEVICE_SELECTOR'] = selector
            print(f"Selected default device: {default_device}")
            print(f"Set ONEAPI_DEVICE_SELECTOR to {selector}")
            return default_device

    def copy_func(self) -> Callable:
        """ Returns the copy method for Dpnp, and ensures array creation is on the selected device. """
        imports = self.imports()  # Import dpnp and dpctl
        dpnp = imports['dpnp']  # Access dpnp from the imports
        device = self.select_device()  # Select the device dynamically

        # Ensure that dpnp.asarray uses the selected device
        return lambda x: dpnp.asarray(x, device=device)

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """ Generates the setup-string, typically for copying data to the selected device. """
        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string for Dpnp, ensuring it runs on the selected device. """
        arg_str = self.arg_str(bench, impl)
        main_exec_str = "__npb_result = __npb_impl({a})".format(a=arg_str)
        return main_exec_str

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the implementations specific to Dpnp for a particular benchmark. """
        # Always load the benchmark from the common dpnp file
        module_pypath = "npbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"].replace('/', '.'),
            m=bench.info["module_name"]
        )
        
        # Keep the module name the same, like `benchmark_name_dpnp`
        module_str = "{m}_dpnp".format(m=module_pypath)
        func_str = bench.info["func_name"]

        ldict = dict()
        try:
            # Load the same implementation regardless of CPU or GPU, and set device later
            exec(f"from {module_str} import {func_str} as impl", ldict)
        except Exception as e:
            print(f"Failed to load the {self.fname} implementation of {func_str}.")
            raise e

        return [(ldict['impl'], 'default')]
