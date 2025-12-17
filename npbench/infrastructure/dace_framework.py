# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import os
import pkg_resources
import traceback

from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Callable, Literal, Sequence, Tuple, Union

dc_float = None
dc_complex_float = None

class DaceFramework(Framework):
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
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """
        if self.fname == "dace_gpu":
            import cupy

            def cp_copy_func(arr):
                darr = cupy.asarray(arr)
                cupy.cuda.stream.get_current_stream().synchronize()
                return darr

            return cp_copy_func
        return super().copy_func()

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in self.info.keys():
            postfix = self.info["postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        ldict = dict()
        # Import DaCe implementation
        try:
            import copy
            import dace
            import dace.data
            import dace.dtypes as dtypes
            from dace.transformation.optimizer import Optimizer
            from dace.transformation.dataflow import MapFusion, Vectorization, MapCollapse
            from dace.transformation.interstate import LoopToMap
            import dace.transformation.auto.auto_optimize as opt

            exec("from {m} import {f} as ct_impl".format(m=module_str, f=func_str))
        except Exception as e:
            print("Failed to load the DaCe implementation.")
            raise (e)

        ##### Experimental: Load strict SDFG
        sdfg_loaded = False
        if self.load_strict:
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            try:
                strict_sdfg = dace.SDFG.from_file(path)
                sdfg_loaded = True
            except Exception:
                pass

        if not sdfg_loaded:
            #########################################################
            # Prepare SDFGs
            base_sdfg, parse_time = util.benchmark("__npb_result = ct_impl.to_sdfg(simplify=False)",
                                                   out_text="DaCe parsing time",
                                                   context=locals(),
                                                   output='__npb_result',
                                                   verbose=False)
            strict_sdfg = copy.deepcopy(base_sdfg)
            strict_sdfg._name = "strict"
            ldict['strict_sdfg'] = strict_sdfg
            _, strict_time = util.benchmark("strict_sdfg.apply_strict_transformations()",
                                            out_text="DaCe Strict Transformations time",
                                            context=locals(),
                                            verbose=False)
            # sdfg_list = [strict_sdfg]
            # time_list = [parse_time[0] + strict_time[0]]
        else:
            ldict['strict_sdfg'] = strict_sdfg
        parse_time = [0]
        sdfg_list = []
        time_list = []

        ##### Experimental: Saving strict SDFG
        if self.save_strict and not sdfg_loaded:
            path = os.path.join(os.getcwd(), 'dace_sdfgs')
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            strict_sdfg.save(path)

        ##########################################################

        try:
            fusion_sdfg = copy.deepcopy(strict_sdfg)
            fusion_sdfg._name = "fusion"
            ldict['fusion_sdfg'] = fusion_sdfg
            _, fusion_time1 = util.benchmark("fusion_sdfg.apply_transformations_repeated([MapFusion])",
                                             out_text="DaCe MapFusion time",
                                             context=locals(),
                                             verbose=False)
            _, fusion_time2 = util.benchmark("fusion_sdfg.apply_strict_transformations()",
                                             out_text="DaCe Strict Transformations time",
                                             context=locals(),
                                             verbose=False)
            sdfg_list.append(fusion_sdfg)
            # time_list.append(time_list[-1] + fusion_time1[0] + fusion_time2[0])
            time_list.append(parse_time[0] + fusion_time1[0] + fusion_time2[0])
        except Exception as e:
            print("DaCe MapFusion failed")
            print(e)
            fusion_sdfg = copy.deepcopy(strict_sdfg)
            ldict['fusion_sdfg'] = fusion_sdfg

        ###########################################################

        def parallelize(sdfg):
            from dace.sdfg import propagation
            try:
                strict_xforms = dace.transformation.simplification_transformations()
            except Exception:
                strict_xforms = None

            for sd in sdfg.all_sdfgs_recursive():
                propagation.propagate_states(sd)
            if strict_xforms:
                sdfg.apply_transformations_repeated([LoopToMap, MapCollapse] + strict_xforms)
            else:
                num = 1
                while num > 0:
                    num = sdfg.apply_transformations_repeated([LoopToMap, MapCollapse])
                    sdfg.simplify()

        try:
            parallel_sdfg = copy.deepcopy(fusion_sdfg)
            parallel_sdfg._name = "parallel"
            ldict['parallel_sdfg'] = parallel_sdfg
            _, ptime1 = util.benchmark("parallelize(parallel_sdfg)",
                                       out_text="DaCe LoopToMap time1",
                                       context=locals(),
                                       verbose=False)
            _, ptime2 = util.benchmark("parallel_sdfg.apply_transformations_repeated([MapFusion])",
                                       out_text="DaCe LoopToMap time2",
                                       context=locals(),
                                       verbose=False)
            sdfg_list.append(parallel_sdfg)
            time_list.append(time_list[-1] + ptime1[0] + ptime2[0])

        except Exception as e:
            print("DaCe LoopToMap failed")
            print(e)
            parallel_sdfg = copy.deepcopy(fusion_sdfg)
            ldict['parallel_sdfg'] = parallel_sdfg

        ###########################################################
        ###### Standalone Test Auto - Opt after strict transformation
        try:

            def autoopt(sdfg, device, symbols):  #, nofuse):
                # # Mark arrays as on the GPU
                # if device == dtypes.DeviceType.GPU:
                #     for k, v in sdfg.arrays.items():
                #         if not v.transient and type(v) == dace.data.Array:
                #             v.storage = dace.dtypes.StorageType.GPU_Global

                # Auto-optimize SDFG
                opt.auto_optimize(auto_opt_sdfg, device, symbols=symbols, use_gpu_storage=True)

            auto_opt_sdfg = copy.deepcopy(strict_sdfg)
            auto_opt_sdfg._name = 'auto_opt'
            ldict['auto_opt_sdfg'] = auto_opt_sdfg
            device = dtypes.DeviceType.GPU if self.info["arch"] == "gpu" else dtypes.DeviceType.CPU

            _, auto_time = util.benchmark(f"autoopt(auto_opt_sdfg, device, symbols = locals())",
                                          out_text="DaCe Auto - Opt",
                                          context=locals(),
                                          verbose=False)

            sdfg_list.append(auto_opt_sdfg)
            time_list.append(time_list[-1] + auto_time[0])

        except Exception as e:
            print("DaCe autoopt failed")
            # print(e)
            # traceback.print_exc()
            auto_opt_sdfg = copy.deepcopy(strict_sdfg)
            ldict['auto_opt_sdfg'] = auto_opt_sdfg

        def vectorize(sdfg, vec_len=None):
            matches = []
            for xform in Optimizer(sdfg).get_pattern_matches(patterns=[Vectorization]):
                matches.append(xform)
            for xform in matches:
                if vec_len:
                    xform.vector_len = vec_len
                xform.apply(sdfg)

        if self.info["arch"] == "gpu":
            def_impl = dace.Config.get('library', 'blas', 'default_implementation')
            if def_impl != "pure":
                dace.Config.set('library', 'blas', 'default_implementation', value='cuBLAS')

        def copy_to_gpu(sdfg):
            opt.apply_gpu_storage(sdfg)
            # for k, v in sdfg.arrays.items():
            #     if not v.transient and isinstance(v, dace.data.Array):
            #         v.storage = dace.dtypes.StorageType.GPU_Global

        if self.info["arch"] == "gpu":
            import cupy as cp

        implementations = []
        for sdfg, t in zip(sdfg_list, time_list):
            ldict['sdfg'] = sdfg
            fe_time = t
            if sdfg._name != 'auto_opt':
                device = dtypes.DeviceType.GPU if self.info["arch"] == "gpu" else dtypes.DeviceType.CPU
                # if self.info["arch"] == "cpu":
                #     # GPUTransform will set GPU schedules by itself
                opt.set_fast_implementations(sdfg, device)
            if self.info["arch"] == "gpu":
                if sdfg._name in ['strict', 'parallel', 'fusion']:
                    _, gpu_time1 = util.benchmark("copy_to_gpu(sdfg)",
                                                  out_text="DaCe GPU transformation time1",
                                                  context=locals(),
                                                  verbose=False)

                    _, gpu_time2 = util.benchmark("sdfg.apply_gpu_transformations()",
                                                  out_text="DaCe GPU transformation time2",
                                                  context=locals(),
                                                  verbose=False)
                    _, gpu_time3 = util.benchmark("sdfg.simplify()",
                                                  out_text="DaCe GPU transformation time3",
                                                  context=locals(),
                                                  verbose=False)
                    # NOTE: to be fair, allow one additional greedy MapFusion after GPU trafos
                    _, gpu_time4 = util.benchmark("sdfg.apply_transformations_repeated(MapFusion)",
                                                  out_text="DaCe GPU transformation time4",
                                                  context=locals(),
                                                  verbose=False)
                    fe_time += gpu_time2[0] + gpu_time3[0] + gpu_time4[0]
                    opt.set_fast_implementations(sdfg, device)
                else:
                    gpu_time1 = [0]
                fe_time += gpu_time1[0]
            try:
                dc_exec, compile_time = util.benchmark("__npb_result = sdfg.compile()",
                                                       out_text="DaCe compilation time",
                                                       context=locals(),
                                                       output='__npb_result',
                                                       verbose=False)
                implementations.append((dc_exec, sdfg._name))
            except Exception as e:
                print("Failed to compile DaCe {a} {s} implementation.".format(a=self.info["arch"], s=sdfg._name))
                print(e)
                traceback.print_exc()
                print("Traceback")
                continue

            fe_time += compile_time[0]

        return implementations

    def params(self, bench: Benchmark, impl: Callable = None):
        return [p for p in bench.info["parameters"]['L'].keys() if p not in bench.info["input_args"]]

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
    
    def set_datatype(self, datatype: Union[Literal['float32'], Literal['float64'], None]):
        # We might get None here if no datatype is specified. This is sad since we cannot know the exact datatype here
        # and we are relying on the fact that frameworks have their default datatypes set to float32.
        super().set_datatype(datatype)
        global dc_float, dc_complex_float
        from dace import float32, float64, complex64, complex128
        dc_float = float64 if datatype == 'float64' else float32
        dc_complex_float = complex128 if datatype == 'float64' else complex64
