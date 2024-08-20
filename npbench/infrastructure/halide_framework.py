# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import os
import pathlib
import tempfile
import shlex
import subprocess
import pkg_resources

from sysconfig import get_paths
from pathlib import Path

from npbench.infrastructure import Benchmark, Framework, utilities as util
from typing import Callable, Sequence, Tuple


class HalideFramework(Framework):
    """ A class for reading and processing framework information. """
    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """
        super().__init__(fname)

    def version(self) -> str:
        """ Returns the framework version. """
        return pkg_resources.get_distribution("halide").version

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """
        module_name = bench.info["module_name"]

        module_pypath = "npbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"].replace('/', '.'),
            m=module_name)
        module_str = "{m}_halide".format(m=module_pypath)
        func_str = bench.info["func_name"]

        benchmark_dir = Path(__file__).parent.parent / "benchmarks" / bench.info["relative_path"]
        halide_cache = benchmark_dir / ".halidecache"
        halide_cache.mkdir(parents=False, exist_ok=True)

        # Load pipeline
        import halide as hl
        exec("from {m} import {f} as ct_impl".format(m=module_str, f=func_str))
        exec("from {m} import input_buffers".format(m=module_str, f=func_str))
        exec("from {m} import set_estimates".format(m=module_str, f=func_str))
        exec('output_buffers = ct_impl(**input_buffers)')
        exec('pipeline = hl.Pipeline(list(output_buffers.values()))')

        # Auto scheduling
        parameters = self.param_str(bench)
        exec('hl.load_plugin("autoschedule_adams2019")')
        exec(f'set_estimates(**input_buffers, **output_buffers, {parameters})')
        exec('pipeline.auto_schedule("Adams2019", hl.get_target_from_environment())')

        # Compile
        object_file_path = halide_cache / f"{module_name}_halide.o"
        python_extension_path = halide_cache / f"{module_name}_halide.py.cpp"
        stmt_html_path = halide_cache / f"{module_name}_halide.html"
        c_source_path = halide_cache / f"{module_name}_halide.cpp"

        generator_options = f'hl.OutputFileType.object: "{object_file_path}", hl.OutputFileType.python_extension: "{python_extension_path}", hl.OutputFileType.stmt_html: "{stmt_html_path}", hl.OutputFileType.c_source: "{c_source_path}"'
        generator_options = "{" + generator_options + "}"
        
        generator_str = f'pipeline.compile_to({generator_options}, list(input_buffers.values()), "{module_name}_halide")'
        exec(generator_str)

        # Compile .so file
        so_file_path = halide_cache / f"{module_name}_halide.so"
        compiler = "g++"
        flags = "-std=c++17 -O3 -pipe -fvisibility=hidden -fvisibility-inlines-hidden -fno-omit-frame-pointer -lz -rdynamic -Wl,-rpath,/usr/local/lib/ -fPIC"
        
        python_include_path = get_paths()["platinclude"]
        includes = f"-I{python_include_path}"
        files = f"-shared {python_extension_path} {object_file_path} -o {so_file_path}"
        so_compile_str = " ".join([compiler, flags, includes, files])
        
        cmd = shlex.split(so_compile_str)
        p = subprocess.Popen(cmd)
        p.wait()

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            bench.info["module_name"] + "_halide", so_file_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        impl = eval("foo.{f}".format(f=bench.info["module_name"] + "_halide"))

        return [(impl, "default")]

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        input_args_str = ", ".join([
            "{b}={a}".format(a=a, b=b)
            for a, b in zip(input_args, bench.info["input_args"])
        ])
        return input_args_str

    def params(self, bench: Benchmark, impl: Callable = None):
        return {
            p: v for p, v in bench.info["parameters"]['paper'].items()
            if p not in bench.info["input_args"]
        }

    def param_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the parameter-string that should be used for calling
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_params = self.params(bench, impl)
        return ", ".join(["{p}={v}".format(p=p, v=v) for p, v in input_params.items()])

