import argparse
import os
import pathlib

from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="numpy")
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'XL'],
                        nargs="?",
                        default='L')
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
    args = vars(parser.parse_args())

    print(args)

    frmwrk = generate_framework(args["framework"])
    numpy = generate_framework("numpy")

    parent_folder = pathlib.Path(__file__).parent.absolute()
    bench_dir = parent_folder.joinpath("bench_info")
    pathlist = pathlib.Path(bench_dir).rglob('*.json')
    for path in pathlist:
        benchname = os.path.basename(path)[:-5]
        bench = Benchmark(benchname)
        lcount = LineCount(bench, frmwrk, numpy)
        lcount.count()
        test = Test(bench, frmwrk, numpy)
        try:
            test.run(args["preset"], args["validate"], args["repeat"],
                     args["timeout"])
        except:
            continue
