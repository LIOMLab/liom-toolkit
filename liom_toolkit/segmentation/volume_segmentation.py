"""Classical 3D brain mask segmentation via SimpleITK watershed."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_fill_holes

from liom_toolkit.segmentation import remove_small_structures


def segment_3d(
    volume: ArrayLike,
    k: int = 5,
    use_log: bool = True,
    threshold_method: str = "otsu",
    fill_holes: bool = True,
) -> NDArray[np.generic]:
    """Segment a 3D brain volume using a watershed algorithm.

    Source:
    https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    Parameters
    ----------
    volume : ArrayLike
        The volume to segment.
    k : int
        The size of the median filter.
    use_log : bool
        Whether to use the log of the volume.
    threshold_method : str
        The threshold method to use. Either "otsu" or "triangle".
    fill_holes : bool
        Whether to fill holes in the mask. Useful for brain segmentation.

    Returns
    -------
    NDArray[np.generic]
        The segmented mask.
    """
    print("Segmenting 3D volume...")
    vol_p = np.copy(volume)
    if use_log:
        vol_p[volume > 0] = np.log(vol_p[volume > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    print("Thresholding image...")
    # Segmenting using an Otsu threshold
    if threshold_method == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif threshold_method == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    print("Applying watershed operations...")
    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    print("Separating foreground and background...")
    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    mask = sitk.GetArrayFromImage(seg)

    # Filling holes and returning the mask
    if fill_holes:
        print("Filling holes...")
        # Fill holes in the mask
        mask = fill_holes_2d_3d(mask)

    print("Removing small structures...")
    # Remove small objects
    return remove_small_structures(vol_p, mask)


def fill_holes_2d_3d(mask: ArrayLike) -> NDArray[np.bool_]:
    """Fill holes in a 2D and 3D mask.

    Source:
    https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    Parameters
    ----------
    mask : ArrayLike
        The mask to fill holes in.

    Returns
    -------
    NDArray[np.bool_]
        The mask with holes filled.
    """
    # Filling holes and returning the mask
    mask = binary_fill_holes(mask)

    # Fill holes (in 2D)
    nx, ny, nz = mask.shape
    for x in range(nx):
        mask[x, :, :] = binary_fill_holes(mask[x, :, :])
    for y in range(ny):
        mask[:, y, :] = binary_fill_holes(mask[:, y, :])
    for z in range(nz):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])

    # Refill holes in 3D (in case some were missed)
    return binary_fill_holes(mask)
