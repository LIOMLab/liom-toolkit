"""Classical 3D brain mask segmentation via SimpleITK watershed."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

# SimpleITK is moved into the [seg] extra (D-01/D-05). The upfront ImportError
# here is the honest signal on an io-only install. The `from e` chain
# preserves the underlying error for debugging (AGENTS §2).
try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError(
        "Please install liom-toolkit[seg] to use the classical segmentation module."
    ) from e

from liom_toolkit.segmentation import remove_small_structures

logger = logging.getLogger(__name__)


def segment_3d(
    volume: NDArray[np.number],
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
    logger.info("Segmenting 3D volume...")
    vol_p = np.copy(volume)
    if use_log:
        vol_p[volume > 0] = np.log(vol_p[volume > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    logger.info("Thresholding image...")
    # Segmenting using an Otsu threshold
    if threshold_method == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif threshold_method == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    logger.info("Applying watershed operations...")
    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    logger.info("Separating foreground and background...")
    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    mask = sitk.GetArrayFromImage(seg)

    # Filling holes and returning the mask
    if fill_holes:
        logger.info("Filling holes...")
        # Fill holes in the mask
        mask = fill_holes_2d_3d(mask)

    logger.info("Removing small structures...")
    # Remove small objects
    return remove_small_structures(vol_p, mask)


def fill_holes_2d_3d(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fill holes in a 2D and 3D mask.

    Vectorized via a single 3D SimpleITK morphological hole-fill call
    (``fullyConnected=True``) that replaces the inherited O(Z+Y+X) per-slice
    scipy cascade (one 3D pass + three per-axis 2D slice passes + a final 3D
    pass). ``fullyConnected=True`` connects diagonally-adjacent foreground
    voxels, matching the scipy default connectivity so the boolean topology
    is identical (gated by a numerical-equivalence regression test asserting
    ``array_equal`` against the old per-slice result).

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
    # One 3D hole-fill call. fullyConnected=True matches the scipy default
    # (diagonal connectivity), so the result is array_equal to the old
    # per-slice cascade on every tested input.
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    filled = sitk.BinaryFillhole(img, fullyConnected=True)
    return sitk.GetArrayFromImage(filled).astype(bool)
