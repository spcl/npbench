# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import os
import pkg_resources
import traceback
import numpy
from dace.transformation.passes.indirect_access_from_nested_sdfg_to_map import IndirectAccessFromNestedSDFGToMap
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
        import dace
        dace.Config.set('cache',value='unique')
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

        def to_dev_tensor(x):
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

        return to_dev_tensor

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


    @staticmethod
    def autotune(_in_sdfg, inputs, dims):
        assert dims >= 0 and dims <= 3
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "NVIDIA" in device_name:
                print("Nvidia GPU detected:", device_name)
                gpu = "nvidia"
            elif "AMD" in device_name:
                print("AMD GPU detected:", device_name)
                gpu = "amd"
        else:
            print("No GPU detected")

        if not (gpu == "nvidia" or gpu == "amd"):
            raise ValueError("Only Nvidia and AMD GPUs are supported.")

        if gpu == "nvidia":
            warp_size = 32
            static_sram = 48*1024
        elif gpu == "amd":
            warp_size = 64
            static_sram = 64*1024

        memory_tiling = [(16,), (32,), (64,), (128,), (256,)]

        def copy_to_gpu(sdfg):
            sdfg.simplify()
            for k, v in sdfg.arrays.items():
                if not v.transient and type(v) == dace.data.Array:
                    v.storage = dace.dtypes.StorageType.GPU_Global
                if v.transient and type(v) == dace.data.Array and v.storage == dace.dtypes.StorageType.Default:
                    v.storage = dace.dtypes.StorageType.GPU_Global

            sdfg.apply_gpu_transformations(validate=True, simplify=True)

        #copy_to_gpu(_in_sdfg)
        if dims == 3:
            thread_coarsening_3D = [(x, y, z) for x, y, z in list(itertools.product(
                [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]))]
            block_sizes_3D = [(x, y, z) for x, y, z in list(itertools.product(
                [1, 2, 4, 8, 16, 32, 64, 128, 256], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32]))
                if x * y * z <= 1024 and (x * y * z) % (warp_size) == 0 and x * y * z >= warp_size]
            thread_coarsening = thread_coarsening_3D
            block_sizes = block_sizes_3D
        elif dims == 2:

            thread_coarsening_2D = [(x, y) for x, y in list(itertools.product(
                [1, 2, 4, 8], [1, 2, 4, 8]))]
            block_sizes_2D = [(x, y) for x, y in list(itertools.product(
                [16, 32, 64, 128, 256], [1, 2, 4, 8, 16]))
                if x * y <= 1024 and (x * y) % (warp_size) == 0 and x * y >= warp_size]

            thread_coarsening = thread_coarsening_2D
            block_sizes = block_sizes_2D
        elif dims == 1:
            thread_coarsening_1D = [(x,) for x in [1, 2, 4, 8, 16, 32, 64]]
            block_sizes_1D = [(x,) for x in [32, 64, 128, 256, 512, 1024]
                if x <= 1024 and (x) % (warp_size) == 0 and x >= warp_size
            ]

            thread_coarsening = thread_coarsening_1D
            block_sizes = block_sizes_1D
        else:
            raise ValueError("Only 1D, 2D and 3D supported.")

        copy_to_gpu(_in_sdfg)
        aopt_sdfg = opt.auto_optimize(sdfg=_in_sdfg, device=dace.dtypes.DeviceType.GPU,
                                      validate=False, validate_all=False, use_gpu_storage=True)
        aopt_sdfg.save("aopt.sdfg")
        #from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
        #aopt_sdfg.apply_transformations_repeated(InlineSDFG)
        #aopt_sdfg.simplify()
        aopt_sdfg.validate()


        dace.Config.set('compiler', 'cpu', 'args', value='-march=native -mtune=native -flto -Ofast -std=c++17 -fPIC')
        dace.Config.set('compiler', 'cuda', 'args', value='-march=native --use_fast_math -O3 -std=c++17 --compiler-options=\"-Ofast\"')
        aopt_sdfg.save("per.sdfg")

        tiled_sdfg, _ = auto_tile_gpu(
            sdfg=aopt_sdfg,
            exhaustive_search=True,
            memory_tiling_parameters=memory_tiling,
            thread_coarsening_parameters=thread_coarsening,
            thread_block_parameters=block_sizes,
            apply_explicit_memory_transfers=[(True, False, True), (True, False, False), (True, True, True), (True, True, False), (False, False, False)],
            apply_remainder_loop=[True],
            inputs=inputs,
            device_schedule = dace.dtypes.ScheduleType.GPU_Device,
            re_apply=False,
            verbose=True,
            timeout=300,
            random_iter=True,
            static_sram_limit=static_sram
        )

        tiled_sdfg = aopt_sdfg
        csdfg = tiled_sdfg.compile()
        csdfg(**copy.deepcopy(inputs))


        return csdfg