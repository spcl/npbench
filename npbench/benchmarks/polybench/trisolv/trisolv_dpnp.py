import dpnp as np

def kernel(L, x, b):
    for i in range(x.shape[0]):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return (L, x, b)