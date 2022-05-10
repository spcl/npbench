import numpy as np
import dace

def blur_cols_transpose(img, alpha: dace.float32):
    res = np.empty(img.shape, dtype=np.float32)
    res[0, :, :] = img[0, :, :]

    for y in range(1, res.shape[0]):
        res[y, :, :] = (1.0 - alpha) * res[y - 1, :, :] + alpha * img[y, :, :]

    for y in range(res.shape[0] - 2, 0):
        res[y, :, :] = (1.0 - alpha) * res[y + 1, :, :] + alpha * res[y, :, :]

    res_T = np.empty((img.shape[1], img.shape[0], 3), dtype=np.float32)
    for y, x in dace.map[0:img.shape[0],0:img.shape[1]]:
        res_T[x, y] = res[y, x]

    return res_T

@dace.program
def iir_blur(image: dace.uint8[2560, 1536, 3], output: dace.uint8[2560, 1536, 3]):
    imagef = image.astype(np.float32)
    alpha = 0.1

    blur_y = blur_cols_transpose(imagef, alpha)
    blur = blur_cols_transpose(blur_y, alpha)

    blur[blur < 0] = 0
    blur[blur > 255] = 255
    output[:,:,:] = blur.astype(np.uint8)
