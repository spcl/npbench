import numpy as np


#pythran export kernel(int32[:,:])
def kernel(path):

    for k in range(path.shape[0]):
        # path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
        for i in range(path.shape[0]):
            path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])
