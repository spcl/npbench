# https://numba.readthedocs.io/en/stable/user/5minguide.html

import torch
import appy

@appy.jit
def go_fast(a):
    trace = torch.zeros(1, dtype=a.dtype)
<<<<<<< HEAD
    #pragma parallel for
=======
    #pragma parallel
>>>>>>> 45c8817 (added go_fast_appy, updated appy_framework to set default device to cuda)
    for i in range(a.shape[0]):
        #pragma atomic
        trace[0] += torch.tanh(a[i, i])
    return a + trace
