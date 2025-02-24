# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import copy
import os
import pkg_resources
from typing import Callable
import itertools

import dace
import dace.data
import dace.transformation.auto.auto_optimize as opt

from dace.transformation.auto_tile.auto_tile_cpu import auto_tile_cpu

import os
import subprocess

import warnings

from npbench.infrastructure import Framework, Benchmark
from dace.transformation.passes.indirect_access_from_nested_sdfg_to_map import (
    IndirectAccessFromNestedSDFGToMap,
)

# Run the shell command and capture the output
class DaceCPUAutoTileFramework(Framework):
    """ A class for reading and processing framework information. """


    @staticmethod
    def get_num_cores():
        command = "grep -m 1 'cpu cores' /proc/cpuinfo | awk '{print $4}'"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Convert the output to an integer
        num_cores = int(result.stdout.strip())

        command = "nproc"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Convert the output to an integer
        hyperthreads = int(result.stdout.strip())

        return num_cores, hyperthreads

    @staticmethod
    def get_numa_nodes():
        command = 'lscpu | awk \'/NUMA node\\(s\\)/ {print $3}\''
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        numa_nodes = int(result.stdout.strip())

        return numa_nodes

    def __init__(self, fname: str, save_strict: bool = False, load_strict: bool = False):
        """ Reads framework information.
        :param fname: The framework name.
        :param save_strict: If True, saves the simplified SDFG.
        :param load_strict: If True, loads the simplified SDFG.
        """

        num_cores, num_threads = DaceCPUAutoTileFramework.get_num_cores()
        #if os.environ['OMP_NUM_THREADS'] is None or (os.environ['OMP_NUM_THREADS'] != str(num_cores) and os.environ['OMP_NUM_THREADS'] != str(num_threads)):
        #    raise ValueError(f"OMP_NUM_THREADS not set correctly, ensure it is set to number of CPU cores {num_cores} or number of hyperthreads {num_threads}, found {os.environ['OMP_NUM_THREADS']}")

        self.save_strict = save_strict
        self.load_strict = load_strict

        warnings.filterwarnings("ignore")

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution("dace").version

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
        return main_exec_str

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

    #thread_coarsening_2D = [(x, y) for x, y in list(itertools.product(
    #    [16, 128, 32, 256, 512, 1024, 2048, 4096], [32, 8, 16, 128]))
    #    if x >= y]
    thread_coarsening_2D = [(x, y) for x, y in list(itertools.product(
        list(reversed([16,32,64,128,256,512,1024,2048,4096,8192])),
        list(reversed([16,32,64,128,256,512,1024,2048,4096,8192])),
        ))
        if x >= y and x * y <= 8192*8192//128]
    memory_tiling = [[(128,), (64,), (32,)],]
    #block_sizes_2D = [(x, y) for x, y in list(itertools.product(
    #    [1, 2, 4, 8, 16], [1, 2, 4, 8, 16]))
    #    if x * y == int(os.environ['OMP_NUM_THREADS'])]

    thread_coarsening_3D = [(x, y, z) for x, y, z in list(itertools.product(
        list(reversed([16,32,64,128,256,512,1024,2048,4096,8192])),
        list(reversed([16,32,64,128,256,512,1024,2048,4096,8192])),
        list(reversed([16,32,64,128,256,512,1024,2048,4096,8192]))))
        if x * y * z <= 8192*8192//128]

    @staticmethod
    def validate_and_pad_params_to_three(params):
        validated_params = []
        for param in params:
            if len(param) < 3:
                padded_param = param + (1,) * (3 - len(param))
                validated_params.append(padded_param)
            elif len(param) == 3:
                validated_params.append(param)
            else:
                raise ValueError(
                    f"Tuple {param} has length greater than 3, which is not allowed."
                )
        return validated_params

    @staticmethod
    def autotune(_in_sdfg, inputs, dims):
        assert dims >= 0 and dims <= 3

        aopt_sdfg = opt.auto_optimize(_in_sdfg, dace.dtypes.DeviceType.CPU)

        for state in aopt_sdfg.states():
            for n in state.nodes():
                if isinstance(n, dace.sdfg.nodes.MapEntry):
                    if n.map.schedule == dace.dtypes.ScheduleType.CPU_Multicore:
                        n.map.schedule = dace.dtypes.ScheduleType.Default
        num_cores, num_threads = DaceCPUAutoTileFramework.get_num_cores()
        numa_nodes = DaceCPUAutoTileFramework.get_numa_nodes()

        IndirectAccessFromNestedSDFGToMap().apply_pass(sdfg=aopt_sdfg, _={})

        from dace.transformation.interstate import InlineSDFG, InlineMultistateSDFG
        aopt_sdfg.apply_transformations_repeated(InlineSDFG)
        aopt_sdfg.simplify()

        for s in aopt_sdfg.states():
            for n in s.nodes():
                if isinstance(n, dace.nodes.MapEntry) and n.map.schedule == dace.ScheduleType.CPU_Multicore:
                    n.map.schedule = dace.ScheduleType.Default

        #aopt_sdfg.save("aopt_sdfg1.sdfg")
        aopt_sdfg.validate()
        aopt_sdfg.save("aopt_sdfg_prep2.sdfg")

        dace.Config.set('compiler', 'cpu', 'args', value='-march=native -mtune=native -flto -Ofast -std=c++17 -fPIC')

        _sdfg = None
        best_total_time = 0.0
        tcount = None
        candidates_tried = 0
        for i, (thread_count, msg) in enumerate([(num_cores, "without hyperthreading")]):
            print(f"Start Autotuning {msg}")
            os.environ['OMP_NUM_THREADS'] = str(thread_count)
            if os.environ['OMP_NUM_THREADS'] != str(thread_count):
                print("Setting OMP_NUM_THREADS failed")
                raise Exception("Setting OMP_NUM_THREADS failed")
            block_sizes_2D = [(x, y) for x, y in list(itertools.product(
                [1,2,4,8,16,32,64], [1,2,4,8,16,32,64]))
                if x * y == num_cores]
            block_sizes_3D = [(x, y, z) for x, y, z in list(itertools.product(
                [1,2,4,8,16,32,64], [1,2,4,8,16,32,64], [1,2,4,8,16,32,64]))
                if x * y * z == num_cores]
            if dims == 3:
                thread_coarsening = DaceCPUAutoTileFramework.thread_coarsening_3D
                block_sizes = block_sizes_3D
            elif dims == 2:
                thread_coarsening = DaceCPUAutoTileFramework.thread_coarsening_2D
                block_sizes = block_sizes_2D
            else:
                raise ValueError("Only 2D and 3D supported.")

            combinations = list(
                itertools.product(
                    DaceCPUAutoTileFramework.memory_tiling,
                    DaceCPUAutoTileFramework.validate_and_pad_params_to_three(
                        thread_coarsening),
                    DaceCPUAutoTileFramework.validate_and_pad_params_to_three(
                        block_sizes),
                    [True],
                )
            )
            tiled_sdfg, d = auto_tile_cpu(
                sdfg=copy.deepcopy(aopt_sdfg),
                exhaustive_search=True,
                memory_tiling_parameters=DaceCPUAutoTileFramework.memory_tiling,
                thread_coarsening_parameters=thread_coarsening,
                thread_block_parameters=block_sizes,
                apply_remainder_loop=[True],
                inputs=inputs,
                re_apply=False,
                verbose=True,
                num_cores=int(os.environ['OMP_NUM_THREADS']),
                timeout=300,
                random_iter=True
            )
            _ct = len(combinations)
            candidates_tried += _ct
            if _sdfg is None:
                for v in d.values():
                    best_total_time += v[2]
                _sdfg = tiled_sdfg
                tcount = thread_count
            else:
                _best_total_time = 0.0
                for v in d.values():
                    _best_total_time += v[2]
                if _best_total_time < best_total_time:
                    best_total_time = _best_total_time
                    tcount = thread_count
                    _sdfg = tiled_sdfg
            if _sdfg is None:
                print("SDFG must have been assigned at this stage")
                raise Exception("SDFG must have been assigned at this stage")
            print(f"End Autotuning {msg}")
            print(f"Autotuning tried {candidates_tried} configurations")

        csdfg = _sdfg.compile()
        csdfg(**copy.deepcopy(inputs))

        return csdfg