import argparse

from multiprocessing import Process
from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

def run_benchmark(benchname, fname, preset, validate, repeat, timeout, skip_existing):
        frmwrk = generate_framework(fname)
        numpy = generate_framework("numpy")
        bench = Benchmark(benchname)
        lcount = LineCount(bench, frmwrk, numpy)
        lcount.count()
        test = Test(bench, frmwrk, numpy)
        test.run(preset, validate, repeat, timeout, skip_existing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-t", "--timeout", type=float, nargs="?", default=10.0)
    parser.add_argument("-d", "--dace", type=util.str2bool, nargs="?", default=True)
    parser.add_argument(
        "-e",
        "--skip-existing",
        type=util.str2bool,
        nargs="?",
        default=False,
        help="If set, do not run the benchmark if an entry with the same mode and framework already exists in the database.",
    )
    args = vars(parser.parse_args())

    benchmarks = [
        'adi', 'arc_distance', 'atax', 'azimint_naive', 'bicg', 'cavity_flow',
        'cholesky2', 'compute', 'doitgen', 'floyd_warshall', 'gemm', 'gemver',
        'gesummv', 'go_fast', 'hdiff', 'jacobi_2d', 'lenet', 'syr2k', 'trmm',
        'vadv'
    ]

    frameworks = ["numpy", "numba"]
    if args['dace']:
        frameworks.append("dace_cpu")

    for benchname in benchmarks:
        for fname in frameworks:
            p = Process(
                target=run_benchmark,
                args=(benchname, fname, args["preset"],
                    args["validate"], args["repeat"], args["timeout"], args["skip_existing"])
            )
            p.start()
            p.join()

    # numpy = generate_framework("numpy")
    # numba = generate_framework("numba")

    # for benchname in benchmarks:
    #     bench = Benchmark(benchname)
    #     for frmwrk in [numpy, numba]:
    #         lcount = LineCount(bench, frmwrk, numpy)
    #         lcount.count()
    #         test = Test(bench, frmwrk, numpy)
    #         try:
    #             test.run(args["preset"], args["validate"], args["repeat"],
    #                      args["timeout"])
    #         except:
    #             continue
