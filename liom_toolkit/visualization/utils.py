import numpy as np


def convert_to_png_for_saving(img: np.ndarray) -> np.ndarray:
    """
    Convert the array to be suitable for PNG saving with skimage.io.imsave.

    :param img: The image to convert
    :type img: np.ndarray
    :return: The converted image
    :rtype: np.ndarray
    """
    normalized_image = (img - np.min(img)) * (
            255.0 / (np.max(img) - np.min(img)))
    normalized_image = normalized_image.astype('uint8')
    return normalized_image
