# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import numpy as np

from PIL import Image
from pathlib import Path

def initialize():
    image_path = Path(__file__).parent.parent / "images" / "gray.png"
    image = np.array(Image.open(image_path), dtype=np.uint8)
    output = np.empty((image.shape[0] - 2, image.shape[1] - 2), dtype=np.uint8)

    return image, output
