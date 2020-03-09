import numpy as np


def crop_from_center(image, ijk_coordinate, width, height, depth):
    i, j, k = ijk_coordinate
    crop = image[i - width // 2: i + int(np.ceil(width / 2)),
                 j - height // 2: j + int(np.ceil(height / 2)),
                 k - depth // 2: k + int(np.ceil(depth / 2))]
    return crop
