import numpy as np
import dace as dc

@dc.program
def blur(image: dc.uint8[2560, 1536], output: dc.uint8[2558, 1534]):
    imagef = image.astype(np.float32)
    blur_x = np.empty((imagef.shape[0], imagef.shape[1] - 2), dtype=np.float32)
    blur_y = np.empty(output.shape, dtype=np.float32)

    for y in range(blur_x.shape[0]):
        for x in range(blur_x.shape[1]):
            blur_x[y, x] = (imagef[y, x] + imagef[y, x + 1] + imagef[y, x + 2]) / 3.0
    
    for y in range(blur_y.shape[0]):
        for x in range(blur_y.shape[1]):
            blur_y[y, x] = (blur_x[y, x] + blur_x[y + 1, x] + blur_x[y + 2, x]) / 3.0

    blur_y[blur_y < 0.0] = 0.0
    blur_y[blur_y > 255.0] = 255.0
    output[:,:] = blur_y.astype(np.uint8)
