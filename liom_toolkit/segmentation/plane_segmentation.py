import os

import numpy as np
from scipy.ndimage import median_filter
from skimage import restoration, filters, morphology
from skimage.filters import frangi, thresholding
from skimage.io import imsave
from skimage.measure import regionprops, label
from skimage.morphology import disk, binary_erosion
from skimage.util import img_as_ubyte


def subtract_background(img: np.ndarray, radius: int = 70) -> np.ndarray:
    """
    Subtract background from image using rolling ball algorithm

    :param img: The image to subtract the background from
    :type img: np.ndarray
    :param radius: The radius of the rolling ball
    :type radius: int
    :return: The background subtracted image
    :rtype: np.ndarray
    """
    normalized_radius = radius // 255
    kernel = restoration.ellipsoid_kernel(
        (radius * 2, radius * 2),
        normalized_radius * 2
    )
    rolling_ball = restoration.rolling_ball(img, radius=radius, kernel=kernel)
    return img - rolling_ball


def frangi_filter(img: np.ndarray, sigma_range: tuple, black_ridges: bool = False) -> np.ndarray:
    """
    Apply the Frangi filter to an image

    :param img: The image to apply the filter to
    :type img: np.ndarray
    :param sigma_range: The range of sigmas to use (start, stop, step)
    :type sigma_range: tuple
    :param black_ridges: Whether to detect black ridges
    :type black_ridges: bool
    :return: The filtered image
    :rtype: np.ndarray
    """
    return frangi(img, sigmas=[x for x in range(*sigma_range)], black_ridges=black_ridges)


def li_threshold_image(img: np.ndarray) -> np.ndarray:
    """
    Apply the Li thresholding algorithm to an image

    :param img: The image to apply the thresholding to
    :type img: np.ndarray
    :return: The thresholded image
    :rtype: np.ndarray
    """
    return img > thresholding.threshold_li(img, initial_guess=np.quantile(img, 0.95))


def sauvola_threshold_image(img: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Apply the Sauvola thresholding algorithm to an image

    :param img: The image to apply the thresholding to
    :type img: np.ndarray
    :param window_size: The size of the window to use for thresholding
    :type window_size: int
    :return: The thresholded image
    :rtype: np.ndarray
    """
    return img > filters.threshold_sauvola(img, window_size=window_size)


def estimate_tissue_mask(img: np.ndarray) -> np.ndarray:
    """
    Estimate the tissue mask from an image. Based on function found here: https://github.com/joe-from-mtl/sbhassisant-2d-3d-registration

    :param img: The image to estimate the mask from
    :type img: np.ndarray
    :return: The tissue mask
    :rtype: np.ndarray
    """
    mask_data = img > 0
    # Get a tissue threshold value
    threshold_tissue = thresholding.threshold_triangle(img[mask_data])

    # Apply threshold
    mask = img > threshold_tissue

    # Filter out noisy segmentation
    mask = median_filter(mask, 5)

    mask = remove_small_structures(img, mask)

    return mask


def remove_small_structures(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove small structures from a mask

    :param img: The image with which the mask was generated
    :type img: np.ndarray
    :param mask: The mask to remove small structures from
    :type mask: np.ndarray
    :return: The mask with small structures removed
    :rtype: np.ndarray
    """
    # Filter out small structures
    img_labels = label(mask)
    props = regionprops(img_labels)

    # Area threshold
    img_size = img.size
    tissue_labels = []
    for this_region in props:
        if this_region.area / img_size >= 0.05:
            tissue_labels.append(this_region.label)
    mask = np.zeros_like(mask)
    for this_label in tissue_labels:
        mask[img_labels == this_label] = 1
    return mask


def erode_mask(mask: np.ndarray, disk_size: int = 30) -> np.ndarray:
    """
    Erode the outer edge of a mask

    :param mask: The mask to erode
    :type mask: np.ndarray
    :param disk_size: The size of the disk to use for erosion
    :type disk_size: int
    :return: The eroded mask
    :rtype: np.ndarray
    """
    return binary_erosion(mask, disk(disk_size))


def segment_2d_image(output_dir: str, image: np.ndarray, name: str, frangi_sigma_range: tuple = (2, 16, 2),
                     frangi_black_ridges: bool = False, local_threshold: bool = False,
                     local_threshold_size: int = 15) -> None:
    """
    Segment 2D images. Finished files are not returned due to memory concerns, but are saved to disk.

    :param output_dir: The directory to save the results to
    :type output_dir: str
    :param image: The image to segment
    :type image: np.ndarray
    :param name: The name of the image
    :type name: str
    :param frangi_sigma_range: The range of sigmas to use for the Frangi filter
    :type frangi_sigma_range: tuple
    :param frangi_black_ridges: Whether to detect black ridges
    :type frangi_black_ridges: bool
    :param local_threshold: Whether to use local thresholding
    :type local_threshold: bool
    :param local_threshold_size: The size of the local thresholding window, must be odd
    :type local_threshold_size: int
    """
    assert local_threshold_size % 2 == 1, 'Local thresholding window size must be odd'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
    cleaned = morphology.remove_small_objects(vessel_mask_raw, min_size=200)

    # Apply mask
    vessel_mask = cleaned * mask

    # Save image
    imsave(output_dir + name + '_mask.tif', img_as_ubyte(mask), check_contrast=False)
    imsave(output_dir + name + '_vessel_mask.tif', img_as_ubyte(vessel_mask), check_contrast=False)
    # Clean memory
    del image, mask, frangi, vessel_mask_raw, vessel_mask, cleaned
