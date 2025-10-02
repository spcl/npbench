# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import time

import numpy as np

from npbench.infrastructure import (Benchmark, Framework, timeout_decorator as tout, utilities as util)
from typing import Any, Callable, Dict, Sequence, Tuple


class Test(object):
    """ A class for testing a framework on a benchmark. """

    def __init__(self, bench: Benchmark, frmwrk: Framework, npfrmwrk: Framework = None):
        self.bench = bench
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk

    def _execute(self, frmwrk: Framework, impl: Callable, impl_name: str, mode: str, bdata: Dict[str, Any], repeat: int,
                 ignore_errors: bool, autotuner: Callable = None, autotuner_name: str = "") -> Tuple[Any, Sequence[float]]:
        report_str = frmwrk.info["full_name"] + " - " + impl_name
        autotuner_report_str = frmwrk.info["full_name"] + " - " + autotuner_name + "(autotune)"
        autotuner_str = ""
        print(report_str, autotuner, autotuner_report_str)

        copy = frmwrk.copy_func()
        setup_str = frmwrk.setup_str(self.bench, impl)
        exec_str = frmwrk.exec_str(self.bench, impl)
        if autotuner is not None:
            autotuner_str = frmwrk.autotune_str(self.bench, impl)

        ldict = {'__npb_impl': impl, '__npb_copy': copy, '__npb_autotune': autotuner, **bdata}

        if autotuner_str != "":
            util.benchmark(autotuner_str, setup_str, autotuner_report_str + " - " + mode, repeat, ldict, '__npb_autotune_result')

        out, timelist = util.benchmark(exec_str, setup_str, report_str + " - " + mode, repeat, ldict,
                                           '__npb_result')

        if out is not None:
            if isinstance(out, (tuple, list)):
                out = list(out)
            else:
                out = [out]
        else:
            out = []
        if "output_args" in self.bench.info.keys():
            num_return_args = len(out)
            num_output_args = len(self.bench.info["output_args"])
            out += [ldict[a] for a in frmwrk.inout_args(self.bench)]
            assert len(out) == num_return_args + num_output_args, "Number of output arguments does not match."
        return out, timelist

    def run(self, preset: str, validate: bool, repeat: int, timeout: float = 2000.0, ignore_errors: bool = True, skip_existing: bool = False):
        """ Tests the framework against the benchmark.
        :param preset: The preset to use for testing (S, M, L, paper).
        :param validate: If true, it validates the output against NumPy.
        :param repeat: The number of repeatitions.
        """
        print("***** Testing {f} with {b} on the {p} dataset *****".format(b=self.bench.bname,
                                                                           f=self.frmwrk.info["full_name"],
                                                                           p=preset))
        bdata = self.bench.get_data(preset)

        # create a database connection
        database = r"npbench.db"
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_results_table)
        else:
            print("Error! cannot create the database connection.")

        if (util.check_entry_exists(conn,
                                   self.bench.info["short_name"],
                                   "main",
                                   self.frmwrk.info["simple_name"],
                                   self.frmwrk.version())
            and skip_existing):
            print(f"Entry already exists in database for mode {self.bench.info['short_name']} and framework {self.frmwrk.info['simple_name']}, skipping measurement.")
            return

        # Run NumPy for validation
        print(validate and self.frmwrk.fname != "numpy" and self.numpy, validate, self.frmwrk.fname, self.numpy)
        if validate and self.frmwrk.fname != "numpy" and self.numpy:
            np_impl, np_impl_name = self.numpy.implementations(self.bench)[0]
            np_out, _ = self._execute(self.numpy, np_impl, np_impl_name, "validation", bdata, 1, ignore_errors)
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

        @tout.exit_after(timeout)
        def first_execution(impl, impl_name, autotuner, autotuner_name):
            return self._execute(self.frmwrk, impl, impl_name, "first/validation", context, 1, ignore_errors, autotuner, autotuner_name)

        bvalues = []
        context = {**bdata, **self.frmwrk.imports()}

        autotuner, autotuner_name = self.frmwrk.autotuner(self.bench)

        for impl, impl_name in self.frmwrk.implementations(self.bench):
            # First execution

            frmwrk_out, _ = first_execution(impl, impl_name, autotuner, autotuner_name)


            # Validation
            valid = True
            if validate and np_out is not None:
                try:
                    if isinstance(frmwrk_out, (tuple, list)):
                        frmwrk_out = [self.frmwrk.copy_back_func()(a) for a in frmwrk_out]
                    else:
                        frmwrk_out = self.frmwrk.copy_back_func()(frmwrk_out)

                    frmwrk_name = self.frmwrk.info["full_name"] + " - " + impl_name
                except Exception as e:
                    frmwrk_name = self.frmwrk.info["full_name"]


                frmwrk_name = self.frmwrk.info["full_name"]

                rtol = 1e-5 if not 'rtol' in self.bench.info else self.bench.info['rtol']
                atol = 1e-8 if not 'atol' in self.bench.info else self.bench.info['atol']
                norm_error = 1e-5 if not 'norm_error' in self.bench.info else self.bench.info['norm_error']
                valid = util.validate(np_out, frmwrk_out, frmwrk_name, rtol=rtol, atol=atol, norm_error=norm_error)
                if valid:
                    print("{} - {} - validation: SUCCESS".format(frmwrk_name, impl_name))
                elif not ignore_errors:
                    raise ValueError("{} did not validate!".format(frmwrk_name))

            # Main execution
            _, timelist = self._execute(self.frmwrk, impl, impl_name, "median", context, repeat, ignore_errors, autotuner, autotuner_name)
            if timelist:
                for t in timelist:
                    bvalues.append(dict(details=impl_name, validated=valid, time=t))

        # create a database connection
        database = r"npbench.db"
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_results_table)
        else:
            print("Error! cannot create the database connection.")

        # Write data
        timestamp = int(time.time())
        for d in bvalues:
            new_d = {
                'timestamp': timestamp,
                'benchmark': self.bench.info["short_name"],
                'kind': kind,
                'domain': domain,
                'dwarf': dwarf,
                'preset': preset,
                'mode': "main",
                'framework': self.frmwrk.info["simple_name"],
                'version': version,
                'details': d["details"],
                'validated': d["validated"],
                'time': d["time"]
            }
            result = tuple(new_d.values())
            util.create_result(conn, util.sql_insert_into_results_table, result)

