# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import json
import numpy as np
import pathlib
import pkg_resources

from npbench.infrastructure import Benchmark
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Literal

np_float = None
np_complex = None

class Framework(object):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        self.fname = fname

        parent_folder = pathlib.Path(__file__).parent.absolute()
        frmwrk_filename = "{f}.json".format(f=fname)
        frmwrk_path = parent_folder.joinpath("..", "..", "framework_info", frmwrk_filename)
        try:
            with open(frmwrk_path) as json_file:
                self.info = json.load(json_file)["framework"]
                # print(self.info)
        except Exception as e:
            print("Framework JSON file {f} could not be opened.".format(f=frmwrk_filename))
            raise (e)

    def version(self) -> str:
        """ Returns the framework version. """
        return pkg_resources.get_distribution(self.fname).version

    def imports(self) -> Dict[str, Any]:
        """ Returns a dictionary any modules and methods needed for running
        a benchmark. """
        return {}

    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """
        return np.copy
    
    def copy_back_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark outputs back to the host. """
        return lambda x: x

    def impl_files(self, bench: Benchmark) -> Sequence[Tuple[str, str]]:
        """ Returns the framework's implementation files for a particular
        benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementation files.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        pymod_path = parent_folder.joinpath("..", "..", "npbench", "benchmarks", bench.info["relative_path"],
                                            bench.info["module_name"] + "_" + self.info["postfix"] + ".py")
        return [(pymod_path, 'default')]

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
        try:
            exec("from {m} import {f} as impl".format(m=module_str, f=func_str), ldict)
        except Exception as e:
            print("Failed to load the {r} {f} implementation.".format(r=self.info["full_name"], f=func_str))
            raise e

        return [(ldict['impl'], 'default')]

    def args(self, bench: Benchmark, impl: Callable = None):
        """ Generates the input arguments that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return [
            "__npb_{pr}_{a}".format(pr=self.info["prefix"], a=a) if a in bench.info["array_args"] else a
            for a in bench.info["input_args"]
        ]

    def mutable_args(self, bench: Benchmark, impl: Callable = None):
        """ Generates the input/output arguments that should be copied during
        the setup.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return ["__npb_{pr}_{a}".format(pr=self.info["prefix"], a=a) for a in bench.info["array_args"]]
    

    def inout_args(self, bench: Benchmark, impl: Callable = None):
        """ Generates the input/output arguments that should be checked during
        validation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return ["__npb_{pr}_{a}".format(pr=self.info["prefix"], a=a) for a in bench.info["output_args"]]
    

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        return ", ".join(input_args)

    def out_arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used during the setup.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        output_args = self.mutable_args(bench, impl)
        return ", ".join(output_args)

    def setup_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the setup-string that should be used before calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        if len(bench.info["array_args"]):
            arg_str = self.out_arg_str(bench, impl)
            copy_args = ["__npb_copy({})".format(a) for a in bench.info["array_args"]]
            return arg_str + " = " + ", ".join(copy_args)
        return "pass"

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        # param_str = self.param_str(bench, impl)
        return "__npb_result = __npb_impl({a})".format(a=arg_str)
    
    def set_datatype(self, datatype: Union[Literal["float32"], Literal["float64"]]):
        """ Sets the datatype for the framework.
        :param datatype: The datatype to set (float32, float64).
        """
        global np_float, np_complex
        if datatype == 'float32':
            np_float = np.float32
            np_complex = np.complex64
        else:
            np_float = np.float64
            np_complex = np.complex128

def generate_framework(fname: str, save_strict: bool = False, load_strict: bool = False) -> Framework:
    """ Generates a framework object with the correct class.
    :param fname: The framework name.
    :param save_strict: (dace_cpu/dace_gpu only) If True, saves the simplified SDFG.
    :param load_strict: (dace_cpu/dace_gpu only) If True, loads the simplified SDFG.
    """

    parent_folder = pathlib.Path(__file__).parent.absolute()
    frmwrk_filename = "{f}.json".format(f=fname)
    frmwrk_path = parent_folder.joinpath("..", "..", "framework_info", frmwrk_filename)
    try:
        with open(frmwrk_path) as json_file:
            info = json.load(json_file)["framework"]
            # print(info)
    except Exception as e:
        print("Framework JSON file {f} could not be opened.".format(f=frmwrk_filename))
        raise (e)

    exec("from npbench.infrastructure import {}".format(info["class"]))
    if fname.startswith('dace'):
        frmwrk = eval(f"{info['class']}(fname, {save_strict}, {load_strict})")
    else:
        frmwrk = eval("{}(fname)".format(info["class"]))
    return frmwrk
