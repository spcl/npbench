import argparse

from npbench.infrastructure import (Benchmark, generate_framework, LineCount, utilities as util)
from npbench.infrastructure.measure import Measurement, Timer, Likwid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--benchmark",
                        type=str,
                        nargs="?",
                        required=True)
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

    # print(args)
    if args["metric"] == "runtime":
        metric = Timer()
    elif args["metric"] == "likwid":
        metric = Likwid()

    bench = Benchmark(args["benchmark"])
    frmwrk = generate_framework(args["framework"])
    numpy = generate_framework("numpy")
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Measurement(bench, frmwrk, metric, numpy)
    test.run(args["preset"], args["validate"], args["repeat"], args["timeout"])
