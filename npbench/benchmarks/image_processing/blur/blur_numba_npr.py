import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True, fastmath=True)
def blur(image, output):
    image = image.astype(np.float32)
    blur_x = np.empty((image.shape[0], image.shape[1] - 2), dtype=np.float32)
    blur_y = np.empty(output.shape, dtype=np.float32)

    filter = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0

    for y in nb.prange(blur_x.shape[0]):
        blur_x[y, :] = np.convolve(image[y, :], filter)[2:-2]
    
    for x in nb.prange(blur_y.shape[1]):
        blur_y[:, x] = np.convolve(blur_x[:, x], filter)[2:-2]

    output[:,:] = np.clip(blur_y, 0.0, 255.0).astype(np.uint8)
