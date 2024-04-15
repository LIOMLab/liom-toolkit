import os

import numpy as np


def fix_even(number: int) -> int:
    """
    Fix even numbers by adding 1

    :param number: The number to fix
    :return: The fixed number
    """
    if number % 2 == 0:
        number += 1
    return number


def clean_dir(directory: str) -> None:
    """
    Remove default files in a directory.

    :param directory: The directory to clean.
    :type directory: str
    """
    if os.path.exists(directory + '.DS_Store'):
        os.remove(directory + '.DS_Store')


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
