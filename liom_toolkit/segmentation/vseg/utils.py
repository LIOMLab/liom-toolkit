"""Utilities for vessel segmentation: CLAHE, patching, metrics, file sorting."""

from __future__ import annotations

import os
from glob import glob
from typing import TYPE_CHECKING

import imageio.v3 as iio
import natsort
import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.util import view_as_windows

if TYPE_CHECKING:
    import torch

Image.MAX_IMAGE_PIXELS = 2_000_000_000  # finite DoS-guard limit (not None — AGENTS §2)


def create_dir(path: str) -> None:
    """Create a directory if it does not exist yet.

    Parameters
    ----------
    path : str
        The path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> list[float]:
    """Calculate metrics between ground truth and prediction.

    Metrics are F1, Recall, Precision, Accuracy, and Jaccard.

    Parameters
    ----------
    y_true : ArrayLike
        Ground truth.
    y_pred : ArrayLike
        Prediction.

    Returns
    -------
    list[float]
        ``[f1, recall, accuracy, jaccard, precision]``.

    Raises
    ------
    ImportError
        If scikit-learn is not installed (re-raised with an actionable message).
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            jaccard_score,
            precision_score,
            recall_score,
        )
    except ImportError as e:
        raise ImportError(
            "Please install scikit-learn (ai extra) to use the vessel "
            "segmentation metrics of the LIOM toolkit."
        ) from e

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_jaccard = jaccard_score(y_true, y_pred)

    return [score_f1, score_recall, score_acc, score_jaccard, score_precision]


def process_image(image: ArrayLike, device: torch.device) -> torch.Tensor:
    """Process an image to present to the U-Net model.

    Parameters
    ----------
    image : ArrayLike
        The image to process.
    device : torch.device
        The device to use.

    Returns
    -------
    torch.Tensor
        The processed image as a float32 tensor on ``device`` with shape
        ``(1, 1, H, W)``.

    Raises
    ------
    ImportError
        If PyTorch is not installed (re-raised with an actionable message).
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Please install PyTorch to use the vessel segmentation module of the LIOM toolkit."
        ) from e
    x = np.expand_dims(image, axis=0)
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    return x.to(device)


# Sort a list of filenames by numerical order
# This is used to order the patches as they are named 0_0_png, 1_0_png,
# 2_0_png ... (instead of 0, 1, 10, 11 ...)
def numeric_filesort(path: str, folder: str = "images", extension: str = "png") -> list[str]:
    """Sort a list of filenames by numerical order.

    Parameters
    ----------
    path : str
        The path to the folder.
    folder : str
        The folder to sort.
    extension : str
        The extension of the files.

    Returns
    -------
    list[str]
        The sorted list of filenames.
    """
    test = sorted(glob(f"{path}/{folder}/*{extension}"))
    return natsort.natsorted(test, reverse=False)


# Add a inferred patch to empty array
def add_patch_to_empty_array(
    inference: NDArray[np.generic],
    pred_y: NDArray[np.generic],
    coords: tuple[int, int],
    stride: int,
    overlap: int,
    size: tuple[int, int],
) -> NDArray[np.generic]:
    """Add an inferred patch to an empty array.

    Parameters
    ----------
    inference : NDArray[np.generic]
        The empty array to add the patch to.
    pred_y : NDArray[np.generic]
        The predicted patch.
    coords : tuple[int, int]
        The coordinates of the patch.
    stride : int
        The stride of the patch.
    overlap : int
        The overlap of the patch.
    size : tuple[int, int]
        The size of the patch.

    Returns
    -------
    NDArray[np.generic]
        The array with the patch added (overlapping regions averaged).
    """
    H = size[0]
    W = size[1]

    patch_x1 = coords[0] * stride
    patch_y1 = coords[1] * stride
    inference[patch_x1 : (patch_x1 + H), patch_y1 : (patch_y1 + W)] += pred_y

    if (coords[1] > 0 or coords[0] > 0) and overlap > 0:
        x1 = patch_x1
        y1 = patch_y1

        # If this is the first row (cannot be the first patch)
        if coords[0] == 0:
            x2 = x1 + H
            y2 = y1 + overlap
            to_add = [(x1, x2, y1, y2)]

        # If this is not the first row
        elif coords[0] > 0:
            # If this is the first column
            if coords[1] == 0:
                x2 = x1 + overlap
                y2 = y1 + W
                to_add = [(x1, x2, y1, y2)]

            # If this is between the first and last columns
            else:
                # This yields 2 rectangles:
                # rec1
                x1a = x1
                x2a = x1 + H
                y1a = y1
                y2a = y1 + overlap + 1

                # rec2
                x1b = x1
                x2b = x1 + overlap
                y1b = y1 + overlap + 1
                y2b = y1 + W
                to_add = [(x1a, x2a, y1a, y2a), (x1b, x2b, y1b, y2b)]

        for rectangle in to_add:
            inference[rectangle[0] : rectangle[1], rectangle[2] : rectangle[3]] = (
                inference[rectangle[0] : rectangle[1], rectangle[2] : rectangle[3]] / 2
            )

    return inference


def crop_image(image: ArrayLike, size: tuple[int, int], stride: int) -> NDArray[np.generic]:
    """Crop an image to a specific size and stride.

    Parameters
    ----------
    image : ArrayLike
        The image to crop.
    size : tuple[int, int]
        The size to crop to.
    stride : int
        The stride to crop with.

    Returns
    -------
    NDArray[np.generic]
        The cropped image.
    """
    # size[0] is the window height (axis 0 / rows), size[1] is the window
    # width (axis 1 / cols) — view_as_windows requires the window shape to
    # match the image axes. The x-axis crop (cols) must pair image.shape[1]
    # with size[1]; the y-axis crop (rows) pairs image.shape[0] with
    # size[0]. Using size[0] for both (the previous code) is correct only
    # for square patches and silently mis-crops for non-square windows.
    to_remove_x = (image.shape[1] - size[1]) % stride
    to_remove_left_x = np.floor(to_remove_x / 2).astype(int)
    to_remove_right_x = np.ceil(to_remove_x / 2).astype(int)

    to_remove_y = (image.shape[0] - size[0]) % stride
    to_remove_left_y = np.floor(to_remove_y / 2).astype(int)
    to_remove_right_y = np.ceil(to_remove_y / 2).astype(int)

    return image[
        to_remove_left_y : image.shape[0] - to_remove_right_y,
        to_remove_left_x : image.shape[1] - to_remove_right_x,
    ]


def create_patches(
    image_path: str,
    size: tuple[int, int] = (256, 256),
    stride: int = 64,
    norm: bool = False,
    norm_params: tuple[float, float] = (10, 0.05),
) -> tuple[list[NDArray[np.generic]], tuple[int, ...], tuple[int, ...], NDArray[np.generic]]:
    """Create patches from an image.

    Parameters
    ----------
    image_path : str
        The path to the image.
    size : tuple[int, int]
        The size of the patches.
    stride : int
        The stride of the patches.
    norm : bool
        Normalize the patches (apply CLAHE).
    norm_params : tuple[float, float]
        The parameters for the normalization: ``(kernel_size, clip_limit)``.

    Returns
    -------
    patches : list[NDArray[np.generic]]
        The list of extracted patches.
    image_shape : tuple[int, ...]
        The shape of the cropped image.
    patch_shape : tuple[int, ...]
        The shape of the patch grid.
    image_clahe : NDArray[np.generic]
        The (optionally CLAHE-processed) cropped image.

    Raises
    ------
    ValueError
        If the input image is all-zero after cropping (cannot normalise).
    """
    patches: list[NDArray[np.generic]] = []

    image = iio.imread(image_path)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = crop_image(image, size, stride)
    # Normalize to uint8 by scaling to [0, 255]. An all-zero image (after
    # crop) has max == 0, which would divide-by-zero and silently produce
    # a zero-filled/NaN array — a wrong-data fallback forbidden by the
    # project correctness rules. Make the failure explicit instead.
    max_val = image.max()
    if max_val == 0:
        raise ValueError(
            "create_patches: input image is all-zero after crop; cannot "
            "normalize. Check the input image path or crop parameters."
        )
    image = (image / max_val * 255).astype(np.uint8)

    image_clahe = apply_clahe(image, norm_params[0], norm_params[1]) if norm else image

    patch = view_as_windows(image_clahe, size, stride)

    patches.extend(patch.reshape(patch.shape[0] * patch.shape[1], patch.shape[2], patch.shape[3]))

    return patches, image.shape, patch.shape, image_clahe


def apply_clahe(image: ArrayLike, kernel_size: int, clip_limit: float) -> NDArray[np.generic]:
    """Apply CLAHE (adaptive histogram equalization) to an image.

    Parameters
    ----------
    image : ArrayLike
        The image to process.
    kernel_size : int
        The kernel size for the CLAHE operation.
    clip_limit : float
        The clip limit for the CLAHE operation.

    Returns
    -------
    NDArray[np.generic]
        The CLAHE-processed image.
    """
    return equalize_adapthist(image, kernel_size=kernel_size, clip_limit=clip_limit, nbins=256)
