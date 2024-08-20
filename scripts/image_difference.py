import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Provide two images.')
parser.add_argument('imageA', help='')
parser.add_argument('imageB', help='')

args = parser.parse_args()

image_a = np.array(Image.open(args.imageA)).astype(np.float32)
image_b = np.array(Image.open(args.imageB)).astype(np.float32)

assert image_a.shape == image_b.shape

diff_image = np.abs(image_b - image_a)
print(diff_image)
diff = diff_image[diff_image > 0]

if diff.shape[0] > 0:
    print("Max abs deviation: ", diff.max())

img = Image.fromarray(diff_image.astype(np.uint8))
img.save("difference.png")

assert (diff <= 5).all()
