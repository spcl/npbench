import argparse
import pathlib
import numpy as np
from npbench import run, str2bool

# Module name
module_name = "stockham_fft"
func_name = "stockham_fft"
domain_name = "kernels"
dwarf_name = "spectral_methods"

# Framework information
finfo = dict(
    kind="microbench",
    domain="kernels",
    dwarf="spectral_methods",
    numpy=dict(module_str="{}_numpy".format(module_name),
               func_str=func_name,
               arch="CPU",
               arg_str="N, R, K, X, np_Y",
               setup_str="np_Y = np.copy(Y)",
               report_str="NumPy",
               out_args=("np_Y", )),
    numba=dict(
        module_str="{}_numba".format(module_name),
        func_str=None,  # special names for Numba
        arch="CPU",
        arg_str="N, R, K, X, nb_Y",
        setup_str="nb_Y = np.copy(Y)",
        report_str="Numba",
        out_args=("nb_Y", )),
    pythran=dict(module_str="{}_pythran".format(module_name),
                 module_path=pathlib.Path(__file__).parent.absolute(),
                 func_str=func_name,
                 arch="CPU",
                 arg_str="N, R, K, X, pt_Y",
                 setup_str="pt_Y = np.copy(Y)",
                 report_str="Pythran",
                 out_args=("pt_Y", )),
    cupy=dict(module_str="{}_cupy".format(module_name),
              func_str=func_name,
              arch="GPU",
              arg_str="N, R, K, gX, gY",
              setup_str="gX, gY = cp.asarray(X), cp.asarray(Y)",
              report_str="CuPy",
              out_args=("gY", )),
    dace_cpu=dict(module_str="{}_dace".format(module_name),
                  func_str=func_name,
                  arch="CPU",
                  arg_str="N=N, R=R, K=K, x=X, y=dc_Y",
                  setup_str="dc_Y = np.copy(Y)",
                  report_str="DaCe CPU",
                  out_args=("dc_Y", )),
    dace_gpu=dict(module_str="{}_dace".format(module_name),
                  func_str="{}".format(func_name),
                  arch="GPU",
                  arg_str="N=N, R=R, K=K, x=gX, y=gY",
                  setup_str="gX, gY = cp.asarray(X), cp.asarray(Y)",
                  report_str="DaCe GPU",
                  out_args=("gY", )))


def rng_complex(shape, rng):
    return (rng.random(shape) + rng.random(shape) * 1j)


def initialize(R, K):
    from numpy.random import default_rng
    rng = default_rng(42)

    N = R**K
    X = rng_complex((N, ), rng)
    Y = np.zeros_like(X, dtype=np.complex128)

    return N, X, Y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="dace_cpu")
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-a",
                        "--append",
                        type=str2bool,
                        nargs="?",
                        default=False)
    args = vars(parser.parse_args())

    # Initialization
    R, K = 4, 10
    N, X, Y = initialize(R, K)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())

    # nb_Y = np.zeros_like(X, dtype=np.complex128)
    # # dace_exec = dc_impl.stockham_fft.compile()
    # dc_Y = np.zeros_like(X, dtype=np.complex128)
    # cp_Y = np.zeros_like(X, dtype=np.complex128)

    # # First execution
    # np_impl.stockham_fft(N, R, K, X, np_Y)
    # nb_impl.stockham_fft(N, R, K, X, nb_Y)
    # # dace_exec(N=N, R=R, K=K, X=X, Y=dc_Y)
    # cp_impl.stockham_fft(N, R, K, X, cp_Y)

    # # Validation
    # assert(np.allclose(np.fft.fft(X), np_Y))
    # assert(np.allclose(np_Y, nb_Y))
    # # assert(np.allclose(np_Y, dc_Y))
    # assert(np.allclose(np_Y, cp_Y))

    # # Benchmark
    # time = timeit.repeat("np.fft.fft(X)",
    #                      setup="pass", repeat=10, number=1, globals=globals())
    # print("NumPy built-in FFT Median time: {}".format(np.median(time)))
    # time = timeit.repeat("np_impl.stockham_fft(N, R, K, X, np_Y)",
    #                      setup="pass", repeat=10, number=1, globals=globals())
    # print("NumPy Median time: {}".format(np.median(time)))
    # time = timeit.repeat("nb_impl.stockham_fft(N, R, K, X, nb_Y)",
    #                      setup="pass", repeat=10, number=1, globals=globals())
    # print("Numba Median time: {}".format(np.median(time)))
    # # time = timeit.repeat("dace_exec(N=N, R=R, K=K, X=X, Y=dc_Y)",
    # #                      setup="pass", repeat=10, number=1, globals=globals())
    # # print("DaCe Median time: {}".format(np.median(time)))
    # time = timeit.repeat("cp_impl.stockham_fft(N, R, K, X, cp_Y)",
    #                      setup="pass", repeat=10, number=1, globals=globals())
    # print("CuPy Median time: {}".format(np.median(time)))
