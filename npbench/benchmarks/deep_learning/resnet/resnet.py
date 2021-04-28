import argparse
import pathlib
import numpy as np
from npbench import run, str2bool

# Module name
module_name = "resnet"
func_name = "resnet_basicblock"
domain_name = "deep_learning"
dwarf_name = "dense_linear_algebra"

# Framework information
finfo = dict(
    kind="microapp",
    domain="deep_learning",
    dwarf="dense_linear_algebra",
    numpy=dict(module_str="{}_numpy".format(module_name),
               func_str=func_name,
               arch="CPU",
               arg_str="input, conv1, conv2, conv3",
               setup_str="pass",
               report_str="NumPy"),
    numba=dict(
        module_str="{}_numba".format(module_name),
        func_str=None,  # special names for Numba
        arch="CPU",
        arg_str="input, conv1, conv2, conv3",
        setup_str="pass",
        report_str="Numba"),
    pythran=dict(module_str="{}_pythran".format(module_name),
                 module_path=pathlib.Path(__file__).parent.absolute(),
                 func_str=func_name,
                 arch="CPU",
                 arg_str="input, conv1, conv2, conv3",
                 setup_str="pass",
                 report_str="Pythran"),
    cupy=dict(module_str="{}_cupy".format(module_name),
              func_str=func_name,
              arch="GPU",
              arg_str="ginput, gconv1, gconv2, gconv3",
              setup_str="ginput, gconv1, gconv2, gconv3 = cp.asarray(input), "
              "cp.asarray(conv1), cp.asarray(conv2), cp.asarray(conv3)",
              report_str="CuPy"),
    dace_cpu=dict(
        module_str="{}_dace".format(module_name),
        func_str=func_name,
        arch="CPU",
        arg_str="input=input, conv1=conv1, conv2=conv2, conv3=conv3, "
        "N=N, W=W, H=H, C1=C1, C2=C2",
        setup_str="pass",
        report_str="DaCe CPU"),
    dace_gpu=dict(
        module_str="{}_dace".format(module_name),
        func_str=func_name,
        arch="GPU",
        arg_str="input=ginput, conv1=gconv1, conv2=gconv2, conv3=gconv3, "
        "N=N, W=W, H=H, C1=C1, C2=C2",
        setup_str=
        "gout, ginput, gconv1, gconv2, gconv3 = cp.asarray(out), cp.asarray(input), "
        "cp.asarray(conv1), cp.asarray(conv2), cp.asarray(conv3)",
        report_str="DaCe GPU",
    ))


def initialize(N, W, H, C1, C2):
    from numpy.random import default_rng
    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float32)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float32)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float32)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float32)
    return (input, conv1, conv2, conv3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="dace_gpu")
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
    # Size constants
    N = 8  #: Batch size
    W = H = 56
    C1 = 256
    C2 = 64
    input, conv1, conv2, conv3 = initialize(N, W, H, C1, C2)
    out = np.ndarray((N, H + 2, W + 2, C2), input.dtype)

    run(args["framework"], module_name, func_name, finfo, args["mode"],
        args["validate"], args["repeat"], args["append"], locals())
