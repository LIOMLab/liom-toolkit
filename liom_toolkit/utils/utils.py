"""Cross-cutting helpers: even-number fixup, directory cleanup, PNG normalization."""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fix_even(number: int) -> int:
    """Fix even numbers by adding 1.

    Parameters
    ----------
    number : int
        The number to fix.

    Returns
    -------
    int
        The fixed number (``number + 1`` when ``number`` is even, else ``number``).
    """
    if number % 2 == 0:
        number += 1
    return number


def clean_dir(directory: str) -> None:
    """Remove default files in a directory.

    Parameters
    ----------
    directory : str
        The directory to clean.
    """
    if os.path.exists(directory + ".DS_Store"):
        os.remove(directory + ".DS_Store")


def convert_to_png_for_saving(img: ArrayLike) -> NDArray[np.uint8]:
    """Convert the array to be suitable for PNG saving with imageio.v3.imwrite.

    Parameters
    ----------
    img : ArrayLike
        The image to convert.

    Returns
    -------
    NDArray[np.uint8]
        The converted image, normalized to ``[0, 255]`` and cast to ``uint8``.
    """
    normalized_image = (img - np.min(img)) * (255.0 / (np.max(img) - np.min(img)))
    normalized_image = normalized_image.astype("uint8")
    return normalized_image
