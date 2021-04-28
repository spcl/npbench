import json
import npbench.utilities as util
import numpy as np
import pathlib
import pkg_resources
import time
import timeit

from typing import Any, Callable, Dict, Sequence, Tuple


timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, __npb_result
"""


class Benchmark(object):
    """ A class for reading and processing benchmark information. """

    def __init__(self, bname: str):
        """ Reads benchmark information.

        :param bname: The benchmark name.
        """

        self.bname = bname
        self.bdata = dict()

        parent_folder = pathlib.Path(__file__).parent.absolute()
        bench_filename = "{b}.json".format(b=bname)
        bench_path = parent_folder.joinpath("..", "bench_info", bench_filename)
        try:
            with open(bench_path) as json_file:
                self.info = json.load(json_file)["benchmark"]
                print(self.info)
        except Exception as e:
            print("Benchmark JSON file {b} could not be opened.".format(
                b=bench_filename))
            raise (e)
    
    def get_data(self, preset: str = 'L') -> Dict[str, Any]:
        """ Initializes the benchmark data.

        :param preset: The data-size preset (S, M, L, XL).
        """

        if preset in self.bdata.keys():
            return self.bdata[preset]
        
        # 1. Create data dictionary
        data = dict()
        # 2. Add parameters to data dictionary
        if preset not in self.info["parameters"].keys():
            raise NotImplementedError("{b} doesn't have a {p} preset.".format(
                b=self.bname, p=preset))
        parameters = self.info["parameters"][preset]
        for k, v in parameters.items():
            data[k] = v
        # 3. Import initialization function
        module_filename = "{m}.py".format(m=self.info["module_name"])
        module_pypath = "npbench.benchmarks.{r}.{m}".format(
            r=self.info["relative_path"].replace('/', '.'),
            m=self.info["module_name"])
        exec_str = "from {m} import {i}".format(
            m=module_pypath, i=self.info["init"]["func_name"])
        try:
            exec(exec_str, data)
        except Exception as e:
            print("Module Python file {m} could not be opened.".format(
                m=module_filename))
            raise (e)
        # 4. Execute initialization
        init_str = "{oargs} = {i}({iargs})".format(
            oargs=",".join(self.info["init"]["output_args"]),
            i=self.info["init"]["func_name"],
            iargs=",".join(self.info["init"]["input_args"]))
        exec(init_str, data)
        
        del data[self.info["init"]["func_name"]]
        self.bdata[preset] = data
        return self.bdata[preset]

class Framework(object):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.

        :param fname: The framework name.
        """

        self.fname = fname

        parent_folder = pathlib.Path(__file__).parent.absolute()
        frmwrk_filename = "{f}.json".format(f=fname)
        frmwrk_path = parent_folder.joinpath("..", "framework_info",
                                             frmwrk_filename)
        try:
            with open(frmwrk_path) as json_file:
                self.info = json.load(json_file)["framework"]
                print(self.info)
        except Exception as e:
            print("Framework JSON file {f} could not be opened.".format(
                f=frmwrk_filename))
            raise (e)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution(self.fname).version
        
    def copy_func(self) -> Callable:
        """ Returns the copy-method that should be used 
        for copying the benchmark arguments. """
        return np.copy

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """ Returns the framework's implementations for a particular benchmark.
        
        :param bench: A benchmark.
        :returns: A list of the benchmark implementations.
        """

        module_pypath = "npbench.benchmarks.{r}.{m}".format(
            r=bench.info["relative_path"].replace('/', '.'),
            m=bench.info["module_name"])
        if "module_postfix" in self.info.keys():
            postfix = self.info["module_postfix"]
        else:
            postfix = self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        ldict = dict()
        try:
            exec("from {m} import {f} as impl".format(m=module_str, f=func_str),
                 ldict)
        except Exception as e:
            print("Failed to load the {r} {f} implementation.".format(
                r=self.info["report_str"], f=func_str))
            raise e 

        return [(ldict['impl'], 'default')]

    def args(self, bench: Benchmark, impl: Callable = None):
        """ Generates the input arguments that should be used for calling
        the benchmark implementation.
        
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return ["__npb_{pr}_{a}".format(pr=self.info["short_name"], a=a)
                for a in bench.info["input_args"]]

    def arg_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the argument-string that should be used for calling
        the benchmark implementation.
        
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        input_args = self.args(bench, impl)
        return ", ".join(input_args)
    
    def setup_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the setup-string that should be used before calling
        the benchmark implementation.

        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        copy_args = ["__npb_copy({})".format(a)
                     for a in bench.info["input_args"]]
        return arg_str + " = " + ", ".join(copy_args)
    
    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.

        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        arg_str = self.arg_str(bench, impl)
        return "__npb_result = __npb_impl({a})".format(a=arg_str)


class BenchFrmwrk(object):
    """ A class for benchmarking a framework. """

    def __init__(self,
                 bench: Benchmark,
                 frmwrk: Framework,
                 npfrmwrk: Framework  = None):
        self.bench = bench
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk

    def _benchmark(self, stmt, setup, out_text, repeat, context):
        ldict = {**context}
        output = timeit.repeat(stmt,
                               setup=setup,
                               repeat=repeat,
                               number=1,
                               globals=ldict)
        res = output[0][1]
        raw_time_list = [a for a, b in output]
        raw_time = np.median(raw_time_list)
        ms_time = util.time_to_ms(raw_time)
        print("{}: {}ms".format(out_text, ms_time))
        return res, raw_time_list

    def _validate(self, ref, val, framework="Unknown"):
        if not isinstance(ref, (tuple, list)):
            ref = [ref]
        if not isinstance(val, (tuple, list)):
            val = [val]
        valid = True
        for r, v in zip(ref, val):
            if not np.allclose(r, v):
                try:
                    import cupy
                    if isinstance(v, cupy.ndarray):
                        relerror = util.relative_error(r, cupy.asnumpy(v))
                    else:
                        relerror = util.relative_error(r, v)
                except Exception:
                    relerror = util.relative_error(r, v)
                if relerror < 1e-10:
                    continue
                valid = False
                print("Relative error: {}".format(relerror))
                # return False
        if not valid:
            print("{} did not validate!".format(framework))
        return valid
    
    def _execute(self, frmwrk, impl, bdata, repeat):
        report_str = frmwrk.info["report_str"]
        try:
            copy = frmwrk.copy_func()
            setup_str = frmwrk.setup_str(self.bench, impl)
            exec_str = frmwrk.exec_str(self.bench, impl)
        except Exception as e:
            print("Failed to load the {}-{} implementation.".format(
                report_str, str(impl)))
            print(e)
            return None, None
        ldict = {'__npb_impl': impl, '__npb_copy': copy, **bdata}
        try:
            out, timelist = self._benchmark(exec_str, setup_str, report_str,
                                            repeat, ldict)
        except Exception as e:
            print("Failed to execute the {}-{} implementation.".format(
                report_str, str(impl)))
            print(e)
            return None, None
        if out is not None:
            if isinstance(out, (tuple, list)):
                out = list(out)
            else:
                out = [out]
        else:
            out = []
        if "out_args" in self.bench.info.keys():
            out += [ldict[a] for a in self.frmwrk.args(self.bench)]
        return out, timelist
        
    
    def run(self, preset, validate, repeat):
        
        bdata = self.bench.get_data(preset)

        # Run NumPy for validation
        if validate and self.frmwrk.fname != "numpy" and self.numpy:
            np_impl = self.numpy.implementations(self.bench)[0]
            np_out, _ = self._execute(self.numpy, np_impl, bdata, 1)
        else:
            validate = False
            np_out = None

        # Extra information
        kind = ""
        if "kind" in self.bench.info.keys():
            kind = self.bench.info["kind"]
        domain = ""
        if "domain" in self.bench.info.keys():
            domain = self.bench.info["domain"]
        dwarf = ""
        if "dwarf" in self.bench.info.keys():
            dwarf = self.bench.info["dwarf"]
        version = self.frmwrk.version()

        bvalues = []
        for impl, impl_name in self.frmwrk.implementations(self.bench):
            # First execution
            frmwrk_out, _ = self._execute(self.frmwrk, impl, bdata, 1)
            # Validation
            valid = True
            if validate and np_out is not None:
                try:
                    valid = self._validate(np_out, frmwrk_out, self.frmwrk.info["report_str"])
                except Exception:
                    print("Failed to run {} validation.".format(self.frmwrk.info["report_str"]))
            # Main execution
            _, timelist = self._execute(self.frmwrk, impl, bdata, repeat)
            if timelist:
                for t in timelist:
                    bvalues.append(dict(details=impl_name,
                                        validated=valid, time=t))

        # create a database connection
        database = r"npbench.db"
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_results_table)
        else:
            print("Error! cannot create the database connection.")

        # if self.frmwrk.fname == "cupy":
        #     version = [p.version for p in pkg_resources.working_set
        #             if p.project_name.startswith("cupy")][0]
        # elif self.frmwrk.fname in ("dace_cpu", "dace_gpu"):
        #     version = pkg_resources.get_distribution("dace").version
        # elif self.frmwrk.fname in ("legate_cpu", "legate_gpu"):
        #     version = pkg_resources.get_distribution("legate.numpy").version
        # else:
        #     version = pkg_resources.get_distribution(self.frmwrk.fname).version
        
        # Write data
        timestamp = int(time.time())
        for d in bvalues:
            new_d = {
                'timestamp': timestamp,
                'benchmark': self.bench.info["short_name"],
                'kind': kind,
                'domain': domain,
                'dwarf': dwarf,
                'mode': "main",
                'framework': self.frmwrk.info["report_str"],
                'version': version,
                'details': d["details"],
                'validated': d["validated"],
                'time': d["time"]
            }
            result = tuple(new_d.values())
            print(result)
            util.create_result(conn, util.sql_insert_into_results_table, result)
