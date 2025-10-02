import argparse
import pathlib
import legate.numpy as np
from npbench import run, str2bool

def kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C
    return (alpha, beta, C, A, B)

def init_data(NI, NJ, NK, datatype):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.empty((NI, NJ), dtype=datatype)
    C[:] = np.random.randn(NI, NJ)
    A = np.empty((NI, NK), dtype=datatype)
    A[:] = np.random.randn(NI, NK)
    B = np.empty((NK, NJ), dtype=datatype)
    B[:] = np.random.randn(NJ, NK)
    return (alpha, beta, C, A, B)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--framework', type=str, nargs='?', default='numpy')
    parser.add_argument('-m', '--mode', type=str, nargs='?', default='main')
    parser.add_argument('-v', '--validate', type=str2bool, nargs='?', default=True)
    parser.add_argument('-r', '--repeat', type=int, nargs='?', default=10)
    parser.add_argument('-a', '--append', type=str2bool, nargs='?', default=False)
    args = vars(parser.parse_args())
    print('Hello from Legate!')
    exit()
    NI, NJ, NK = (2048, 2048, 2048)
    alpha, beta, C, A, B = init_data(NI, NJ, NK, np.float64)
    lg_C = orig_np.copy(C)
    kernel(alpha, beta, lg_C, A, B)
    events = [papi_events.PAPI_DP_OPS, papi_events.PAPI_VEC_DP]
    papi_high.start_counters(events)
    lg_counters = [0] * len(events)

    def update_counters(old, new):
        for i in range(len(old)):
            old[i] += new[i]
        return (old, new)
    time = timeit.repeat('papi_high.read_counters();kernel(alpha, beta, lg_C, A, B);update_counters(lg_counters, papi_high.read_counters());', setup='lg_C = orig_np.copy(C);', repeat=20, number=1, globals=globals())
    print('Numpy median time: {} ({} DP FLOPS)'.format(np.median(time), lg_counters[0] / 20 / np.median(time)))