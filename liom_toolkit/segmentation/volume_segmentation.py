import SimpleITK as sitk
import ants
import numpy as np
from scipy.ndimage import binary_fill_holes

from liom_toolkit.segmentation import remove_small_structures


def segment_3d_brain(volume: ants.ANTsImage, k: int = 5, use_log: bool = True,
                     threshold_method: str = "otsu") -> np.ndarray:
    """
    Segment a 3D brain volume using a watershed algorithm.
    Source: https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    :param volume: The volume to segment.
    :type volume: ants.ANTsImage
    :param k: The size of the median filter.
    :type k: int
    :param use_log: Whether to use the log of the volume.
    :type use_log: bool
    :param threshold_method: The threshold method to use. Either "otsu" or "triangle".
    :type threshold_method: str
    :return: The segmented mask
    :rtype: np.ndarray
    """
    np_volume = volume.numpy()
    vol_p = np.copy(np_volume)
    if use_log:
        vol_p[np_volume > 0] = np.log(vol_p[np_volume > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    # Segmenting using an Otsu threshold
    if threshold_method == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif threshold_method == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    # Filling holes and returning the mask
    mask = fill_holes_2d_3d(sitk.GetArrayFromImage(seg))

    # Remove small objects
    mask = remove_small_structures(vol_p, mask)

    return mask


def fill_holes_2d_3d(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a 2D and 3D mask.
    Source: https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    :param mask: The mask to fill holes in.
    :type mask: np.ndarray
    :return: The mask with holes filled.
    :rtype: np.ndarray
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
    mask = binary_fill_holes(mask)
    return mask
