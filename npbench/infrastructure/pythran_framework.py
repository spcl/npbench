# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import os
import pathlib
import tempfile

from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Callable, Sequence, Tuple


class PythranFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        parent_folder = pathlib.Path(__file__).parent.absolute()
        pymod_path = parent_folder.joinpath("..", "..", "npbench", "benchmarks", bench.info["relative_path"],
                                            bench.info["module_name"] + "_pythran.py")
        tmpdir = tempfile.TemporaryDirectory()
        somod_path = os.path.join(tmpdir.name, bench.info["module_name"] + "_pythran.so")

        compile_str = ("os.system(\"pythran -DUSE_XSIMD -fopenmp -march=native " +
                       "-ffast-math {pp} -o {sp}\")".format(pp=pymod_path, sp=somod_path))
        try:
            _, compile_time = util.benchmark(compile_str, out_text="Pythran compilation time", context=globals())
            fe_time = compile_time[0]
            # Taken from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            import importlib.util
            spec = importlib.util.spec_from_file_location(bench.info["module_name"] + "_pythran", somod_path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            ct_impl = eval("foo.{f}".format(f=bench.info["func_name"]))
        except Exception as e:
            print("Failed to load the Pythran implementation.")
            raise (e)

        return [(ct_impl, 'default')]
