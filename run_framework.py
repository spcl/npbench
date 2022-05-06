import argparse
import os
import pathlib

from multiprocessing import Process
from npbench.infrastructure import (Benchmark, generate_framework, LineCount, utilities as util)
from npbench.infrastructure.measure import Measurement, Timer


def run_benchmark(benchname, fname, preset, validate, repeat, timeout):
    timer = Timer()
    
    frmwrk = generate_framework(fname)
    numpy = generate_framework("numpy")
    bench = Benchmark(benchname)
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Measurement(bench, frmwrk, timer, numpy)
    test.run(preset, validate, repeat, timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="numpy")
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper'],
                        nargs="?",
                        default='S')
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t",
                        "--timeout",
                        type=float,
                        nargs="?",
                        default=200.0)
    parser.add_argument("-M", "--metric", type=str, nargs="?", choices=['runtime', 'likwid'], default="runtime")    
    args = vars(parser.parse_args())


    parent_folder = pathlib.Path(__file__).parent.absolute()
    bench_dir = parent_folder.joinpath("bench_info")
    pathlist = pathlib.Path(bench_dir).rglob('*.json')
    benchnames = [os.path.basename(path)[:-5] for path in pathlist]
    benchnames.sort()
    for benchname in benchnames:
        p = Process(
            target=run_benchmark,
            args=(benchname, args["framework"], args["preset"],
                  args["validate"], args["repeat"], args["timeout"], args["metric"])
        )
        p.start()
        p.join()
