import argparse

from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

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
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a",
                        "--append",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    args = vars(parser.parse_args())

    print(args)

    bench = Benchmark(args["benchmark"])
    frmwrk = generate_framework(args["framework"])
    numpy = generate_framework("numpy")
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run('L', args["validate"], args["repeat"])
