import numpy as np
import numba as nb

@nb.jit(nopython=True, parallel=True, fastmath=True)
def hist(image, output):
    image = image.astype(np.float32)
    gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]

    Cr = (image[:,:,0] - gray) * 0.713 + 128
    Cb = (image[:,:,2] - gray) * 0.564 + 128

    binned_gray = np.clip(gray, 0, 255).astype(np.uint8)

    # [0, ..., 256] where 256 is the rightmost edge
    hist, _ = np.histogram(binned_gray, bins=np.arange(257))

    npixels = gray.shape[0] * gray.shape[1]
    density = np.empty((256,), dtype=np.float32)
    density[:] = hist / npixels

    cdf = np.cumsum(density)

    eq = np.empty_like(binned_gray)
    for y in nb.prange(eq.shape[0]):
        for x in nb.prange(eq.shape[1]):
            eq[y, x] = cdf[binned_gray[y, x]]
    
    eq = eq * 255.0
    eq = np.clip(eq, 0, 255)

    red = eq + (Cr - 128.0) * 1.4
    red = np.clip(red, 0, 255)

    green = eq - 0.343 * (Cb - 128.0) - 0.711 * (Cr - 128.0)
    green = np.clip(green, 0, 255)

    blue = eq + 1.765 * (Cb - 128.0)
    blue = np.clip(blue, 0, 255)

    output[:,:,0] = red.astype(np.uint8)
    output[:,:,1] = green.astype(np.uint8)
    output[:,:,2] = blue.astype(np.uint8)
