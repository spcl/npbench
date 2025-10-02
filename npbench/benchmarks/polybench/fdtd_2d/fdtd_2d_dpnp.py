import dpnp as np

def kernel(TMAX, ex, ey, hz, _fict_):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        np.subtract(ey[1:, :], 0.5 * np.subtract(hz[1:, :], hz[:-1, :], out=hz[1:, :].copy()), out=ey[1:, :])
        np.subtract(ex[:, 1:], 0.5 * np.subtract(hz[:, 1:], hz[:, :-1], out=hz[:, 1:].copy()), out=ex[:, 1:])
        np.subtract(hz[:-1, :-1], 0.7 * (np.subtract(ex[:-1, 1:], ex[:-1, :-1], out=ex[:-1, 1:].copy()) + np.subtract(ey[1:, :-1], ey[:-1, :-1], out=ey[1:, :-1].copy())), out=hz[:-1, :-1])
    return (TMAX, ex, ey, hz, _fict_)