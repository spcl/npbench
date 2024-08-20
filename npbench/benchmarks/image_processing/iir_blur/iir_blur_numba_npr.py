import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True, fastmath=True)
def blur_cols_transpose(img, alpha):
    blur = np.empty(img.shape, dtype=np.float32)
    blur[0, :, :] = img[0, :, :]

    for y in nb.prange(1, blur.shape[0]):
        blur[y, :, :] = (1.0 - alpha) * blur[y - 1, :, :] + alpha * img[y, :, :]

    for y in nb.prange(blur.shape[0] - 2, 0):
        blur[y, :, :] = (1.0 - alpha) * blur[y + 1, :, :] + alpha * blur[y, :, :]

    blur_T = np.transpose(blur, axes=(1, 0, 2))
    return blur_T

@nb.jit(nopython=True, parallel=True, fastmath=True)
def iir_blur(image, output):
    image = image.astype(np.float32)
    alpha = 0.1

    blur_y = blur_cols_transpose(image, alpha)
    blur = blur_cols_transpose(blur_y, alpha)

    output[:,:,:] = np.clip(blur, 0, 255).astype(np.uint8)
