"""Cross-cutting helpers: even-number fixup, directory cleanup, PNG normalization."""

from __future__ import annotations

import pathlib

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
    if pathlib.Path(directory + ".DS_Store").exists():
        pathlib.Path(directory + ".DS_Store").unlink()


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
        A constant image (``max == min``) returns an all-zero array of the
        same shape rather than dividing by zero.
    """
    img = np.asarray(img)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        # Constant image: division would produce inf -> NaN -> 0 via
        # implementation-defined uint8 cast. Return an explicit all-zero
        # array instead (a constant image has no contrast to normalize).
        return np.zeros_like(img, dtype=np.uint8)
    normalized_image = (img - min_val) * (255.0 / (max_val - min_val))
    return normalized_image.astype(np.uint8)
