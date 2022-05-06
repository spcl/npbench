# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import time

from npbench.infrastructure import (Benchmark, Framework, timeout_decorator as
                                    tout, utilities as util)

from npbench.infrastructure.measure.metric import Metric
from npbench.infrastructure.measure.validate import validate as validator


class Measurement(object):
    """ A class for testing a framework on a benchmark. """
    def __init__(self,
                 bench: Benchmark,
                 frmwrk: Framework,
                 metric: Metric,
                 npfrmwrk: Framework = None):
        self.bench = bench
        self.metric = metric
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk

    def run(self,
            preset: str,
            validate: bool,
            repeat: int,
            timeout: float = 200.0):
        """ Tests the framework against the benchmark.
        :param preset: The preset to use for testing (S, M, L, paper).
        :param validate: If true, it validates the output against NumPy.
        :param repeat: The number of repeatitions.
        """
        print("***** Testing {f} with {b} on the {p} dataset *****".format(b=self.bench.bname, f=self.frmwrk.info["full_name"], p=preset))

        bdata = self.bench.get_data(preset)

        # Run NumPy for validation
        if validate and self.frmwrk.fname != "numpy" and self.numpy:
            np_impl, np_impl_name = self.numpy.implementations(self.bench)[0]
            np_out, _ = self.metric.execute(bench=self.bench, frmwrk=self.numpy, impl=np_impl, impl_name=np_impl_name, mode="validation", bdata=bdata, repeat=1)
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
        def first_execution(impl, impl_name):
            return self.metric.execute(bench=self.bench, frmwrk=self.frmwrk, impl=impl, impl_name=impl_name, mode="first/validation", bdata=context, repeat=1)

        bvalues = []
        context = {**bdata, **self.frmwrk.imports()}
        for impl, impl_name in self.frmwrk.implementations(self.bench):
            # First execution
            try:
                frmwrk_out, _ = first_execution(impl, impl_name)
            except KeyboardInterrupt:
                print("Implementation \"{}\" timed out.".format(impl_name),
                      flush=True)
                continue
            except Exception:
                continue

            # Validation
            valid = True
            if validate and np_out is not None:
                try:
                    valid = validator(np_out, frmwrk_out,
                                          self.frmwrk.info["full_name"])
                    if valid:
                        print("{} - {} - validation: SUCCESS".format(
                            self.frmwrk.info["full_name"], impl_name))
                except Exception:
                    print("Failed to run {} validation.".format(
                        self.frmwrk.info["full_name"]))
            # Main execution
            _, timelist = self.metric.execute(bench=self.bench, frmwrk=self.frmwrk, impl=impl, impl_name=impl_name, mode="median", bdata=context, repeat=repeat)
            if timelist:
                for t in timelist:
                    bvalues.append(
                        dict(details=impl_name, validated=valid, time=t))

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
            # print(result)
            util.create_result(conn, util.sql_insert_into_results_table,
                               result)
