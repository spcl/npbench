import numpy as np


# Numerically-stable version of softmax
# pythran export softmax(float32[:,:,:,:])
def softmax(x):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.max(x, axis=-1)[:, :, :, np.newaxis]
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.sum(tmp_out, axis=-1)[:, :, :, np.newaxis]
    return tmp_out / tmp_sum
