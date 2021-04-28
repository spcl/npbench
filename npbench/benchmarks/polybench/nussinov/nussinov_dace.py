import numpy as np
import dace as dc

N = dc.symbol('N', dtype=dc.int64)


@dc.program
def match(b1: dc.int32, b2: dc.int32):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


@dc.program
def kernel(seq: dc.int32[N]):

    table = np.zeros((N, N), np.int32)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i, j] = np.maximum(table[i, j], table[i, j - 1])
            if i + 1 < N:
                table[i, j] = np.maximum(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i, j] = np.maximum(
                        table[i, j],
                        table[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    table[i, j] = np.maximum(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = np.maximum(table[i, j],
                                         table[i, k] + table[k + 1, j])

    return table
