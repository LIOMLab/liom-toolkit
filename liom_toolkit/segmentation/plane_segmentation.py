"""Classical 2D vessel segmentation: Frangi filtering, thresholding, and mask cleanup."""

from __future__ import annotations

import imageio.v3 as iio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter
from skimage import filters, morphology, restoration
from skimage.filters import frangi, thresholding
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion
from skimage.util import img_as_ubyte


def subtract_background(img: ArrayLike, radius: int = 70) -> NDArray[np.float64]:
    """Subtract background from an image using the rolling ball algorithm.

    Parameters
    ----------
    img : ArrayLike
        The image to subtract the background from.
    radius : int
        The radius of the rolling ball.

    Returns
    -------
    NDArray[np.float64]
        The background subtracted image.
    """
    normalized_radius = radius // 255
    kernel = restoration.ellipsoid_kernel((radius * 2, radius * 2), normalized_radius * 2)
    rolling_ball = restoration.rolling_ball(img, radius=radius, kernel=kernel)
    return img - rolling_ball


def frangi_filter(
    img: ArrayLike,
    sigma_range: tuple[int, int, int],
    black_ridges: bool = False,
) -> NDArray[np.float64]:
    """Apply the Frangi filter to an image.

    Parameters
    ----------
    img : ArrayLike
        The image to apply the filter to.
    sigma_range : tuple[int, int, int]
        The range of sigmas to use (start, stop, step).
    black_ridges : bool
        Whether to detect black ridges.

    Returns
    -------
    NDArray[np.float64]
        The filtered image.
    """
    return frangi(img, sigmas=list(range(*sigma_range)), black_ridges=black_ridges)


def li_threshold_image(img: ArrayLike) -> NDArray[np.bool_]:
    """Apply the Li thresholding algorithm to an image.

    Parameters
    ----------
    img : ArrayLike
        The image to apply the thresholding to.

    Returns
    -------
    NDArray[np.bool_]
        The thresholded image.
    """
    return img > thresholding.threshold_li(img, initial_guess=np.quantile(img, 0.95))


def sauvola_threshold_image(img: ArrayLike, window_size: int = 15) -> NDArray[np.bool_]:
    """Apply the Sauvola thresholding algorithm to an image.

    Parameters
    ----------
    img : ArrayLike
        The image to apply the thresholding to.
    window_size : int
        The size of the window to use for thresholding.

    Returns
    -------
    NDArray[np.bool_]
        The thresholded image.
    """
    return img > filters.threshold_sauvola(img, window_size=window_size)


def estimate_tissue_mask(img: ArrayLike) -> NDArray[np.bool_]:
    """Estimate the tissue mask from an image.

    Based on a function from
    https://github.com/joe-from-mtl/sbhassisant-2d-3d-registration.

    Parameters
    ----------
    img : ArrayLike
        The image to estimate the mask from.

    Returns
    -------
    NDArray[np.bool_]
        The tissue mask.
    """
    mask_data = img > 0
    # Get a tissue threshold value
    threshold_tissue = thresholding.threshold_triangle(img[mask_data])

    # Apply threshold
    mask = img > threshold_tissue

    # Filter out noisy segmentation
    mask = median_filter(mask, 5)

    return remove_small_structures(img, mask)


def remove_small_structures(img: ArrayLike, mask: ArrayLike) -> NDArray[np.generic]:
    """Remove small structures from a mask.

    Parameters
    ----------
    img : ArrayLike
        The image with which the mask was generated.
    mask : ArrayLike
        The mask to remove small structures from.

    Returns
    -------
    NDArray[np.generic]
        The mask with small structures removed.
    """
    # Filter out small structures
    img_labels = label(mask)
    props = regionprops(img_labels)

    # Area threshold
    img_size = img.size
    tissue_labels = [
        this_region.label for this_region in props if this_region.area / img_size >= 0.05
    ]
    mask = np.zeros_like(mask)
    for this_label in tissue_labels:
        mask[img_labels == this_label] = 1
    return mask


def erode_mask(mask: ArrayLike, disk_size: int = 30) -> NDArray[np.generic]:
    """Erode the outer edge of a mask.

    Parameters
    ----------
    mask : ArrayLike
        The mask to erode.
    disk_size : int
        The size of the disk to use for erosion.

    Returns
    -------
    NDArray[np.generic]
        The eroded mask.
    """
    return erosion(mask, disk(disk_size))


def segment_2d_image(
    output_dir: str,
    image: ArrayLike,
    name: str,
    frangi_sigma_range: tuple[int, int, int] = (2, 16, 2),
    frangi_black_ridges: bool = False,
    local_threshold: bool = False,
    local_threshold_size: int = 15,
) -> None:
    """Segment a 2D image and save the results to disk.

    Finished files are not returned due to memory concerns, but are saved to
    disk.

    Parameters
    ----------
    output_dir : str
        The directory to save the results to.
    image : ArrayLike
        The image to segment.
    name : str
        The name of the image.
    frangi_sigma_range : tuple[int, int, int]
        The range of sigmas to use for the Frangi filter.
    frangi_black_ridges : bool
        Whether to detect black ridges.
    local_threshold : bool
        Whether to use local thresholding.
    local_threshold_size : int
        The size of the local thresholding window, must be odd.

    Raises
    ------
    ValueError
        If ``local_threshold_size`` is even.
    """
    if local_threshold_size % 2 == 0:
        raise ValueError(f"Local thresholding window size must be odd, got {local_threshold_size}")

    # Overwrite-safe output directory creation via the symlink-aware
    # create_directory helper from utils.zarr_writer: a second call into an
    # existing output_dir shutil.rmtree's the directory then recreates it,
    # eliminating the FileExistsError race on re-run and clearing stale
    # _mask.tif / _vessel_mask.tif files from a previous run. Function-scope
    # import avoids a circular import with utils.zarr_writer at module load
    # time, matching the stats.py / conversion.py:save_zarr pattern.
    from pathlib import Path

    from liom_toolkit.utils.zarr_writer import create_directory

    create_directory(Path(output_dir), overwrite=True)

    # Create full mask
    mask = estimate_tissue_mask(image)

    # Apply Frangi filter
    frangi = frangi_filter(image, frangi_sigma_range, frangi_black_ridges)

    # Apply threshold
    if local_threshold:
        vessel_mask_raw = sauvola_threshold_image(frangi, local_threshold_size)
    else:
        vessel_mask_raw = li_threshold_image(frangi)

    # Cleanup small structures
    cleaned = morphology.remove_small_objects(vessel_mask_raw, max_size=200)

    # Apply mask
    vessel_mask = cleaned * mask

    # Save image. Use Path / operator so a caller passing output_dir without a
    # trailing separator still writes inside the created directory (string
    # concatenation would write to the parent directory).
    iio.imwrite(str(Path(output_dir) / f"{name}_mask.tif"), img_as_ubyte(mask))
    iio.imwrite(str(Path(output_dir) / f"{name}_vessel_mask.tif"), img_as_ubyte(vessel_mask))
    # Clean memory
    del image, mask, frangi, vessel_mask_raw, vessel_mask, cleaned
