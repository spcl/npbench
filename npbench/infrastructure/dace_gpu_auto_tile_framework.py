# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import os
import pkg_resources
import traceback
import numpy
from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Callable, Sequence, Tuple
import itertools

import copy
import dace
import dace.data
import dace.dtypes as dtypes
from dace.transformation.optimizer import Optimizer
from dace.transformation.dataflow import MapFusion, Vectorization, MapCollapse
from dace.transformation.interstate import LoopToMap
import dace.transformation.auto.auto_optimize as opt

from dace.transformation.auto_tile.auto_tile_gpu import auto_tile_gpu


class DaceGPUAutoTileFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str, save_strict: bool = False, load_strict: bool = False):
        """ Reads framework information.
        :param fname: The framework name.
        :param save_strict: If True, saves the simplified SDFG.
        :param load_strict: If True, loads the simplified SDFG.
        """

        self.save_strict = save_strict
        self.load_strict = load_strict

        import warnings
        warnings.filterwarnings("ignore")
        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution("dace").version

    def copy_func(self) -> Callable:
        """Returns a method to copy the benchmark arguments to Triton-compatible tensors."""
        import torch

        def to_triton_tensor(x):
            if isinstance(x, numpy.ndarray):
                # Convert NumPy array to Torch tensor on GPU
                return torch.tensor(x, device="cuda")
            elif isinstance(x, torch.Tensor):
                # Move existing Torch tensor to GPU if not already there
                return x.to("cuda")
            else:
                raise TypeError(
                    "Input should be either a NumPy array or a Torch tensor."
                )

        return to_triton_tensor

    def setup_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """Generates the setup string to use before calling the benchmark implementation.
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

    def imports(self):
        """Import Triton-specific modules."""
        import torch
        return {"torch": torch}

    def params(self, bench: Benchmark, impl: Callable = None):
        return [p for p in bench.info["parameters"]['L'].keys() if p not in bench.info["input_args"]]

    def exec_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """Generates the execution string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: Execution command string.
        """

        arg_str = self.arg_str(bench, impl)
        main_exec_str = f"__npb_result = __npb_impl({arg_str})"
        sync_str = "torch.cuda.synchronize()"
        return main_exec_str + "; " + sync_str

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        params = self.params(bench, impl)
        input_args_str = ", ".join(["{b}={a}".format(a=a, b=b) for a, b in zip(input_args, bench.info["input_args"])])
        params_str = ", ".join(["{a}={a}".format(a=a) for a in params])
        return ", ".join((input_args_str, params_str))

    def param_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the parameter-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_params = self.params(bench, impl)
        return ", ".join(["{p}={p}".format(p=p) for p in input_params])

    def autotune_str(self, bench: Benchmark, impl: Callable = None) -> str:
        """Generates the execution string to call the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        :returns: Execution command string.
        """

        arg_str = self.arg_str(bench, impl)
        autotune_str = f"__npb_autotune_result =__npb_autotune({arg_str})"
        return autotune_str

    thread_coarsening_2D = [(x, y) for x, y in list(itertools.product(
        [1, 2, 4, 8], [1, 2, 4, 8])) if x >= y]
    block_sizes_2D = [(x, y) for x, y in list(itertools.product(
        [16, 32, 64, 128, 256], [1, 2, 4, 8, 16]))
        if x * y <= 1024 and (x * y) % (32) == 0 and x * y >= 32 and x * y <= 512]
    memory_tiling = [(32,)]

    @staticmethod
    def autotune(_in_sdfg, inputs):
        def copy_to_gpu(sdfg):
            for k, v in sdfg.arrays.items():
                if not v.transient and isinstance(v, dace.data.Array):
                    v.storage = dace.dtypes.StorageType.GPU_Global

        copy_to_gpu(_in_sdfg)

        aopt_sdfg = opt.auto_optimize(_in_sdfg, dace.dtypes.DeviceType.GPU)

        tiled_sdfg, _ = auto_tile_gpu(
            sdfg=aopt_sdfg,
            exhaustive_search=True,
            memory_tiling_parameters=DaceGPUAutoTileFramework.memory_tiling,
            thread_coarsening_parameters=DaceGPUAutoTileFramework.thread_coarsening_2D,
            thread_block_parameters=DaceGPUAutoTileFramework.block_sizes_2D,
            apply_explicit_memory_transfers=[(True, False, True), (True, False, False), (False, False, False)],
            apply_remainder_loop=[False],
            inputs=inputs,
            device_schedule = dace.dtypes.ScheduleType.GPU_Device,
            re_apply=False,
            verbose=True,
        )

        return tiled_sdfg