import os

import numpy as np
from scipy.ndimage import median_filter
from skimage import restoration, img_as_ubyte, filters, morphology
from skimage.filters import frangi, thresholding
from skimage.io import imsave
from skimage.measure import regionprops, label
from skimage.morphology import disk, binary_erosion


def subtract_background(img, radius=70):
    """
    Subtract background from image using rolling ball algorithm
    :param img: The image to subtract the background from
    :param radius: The radius of the rolling ball
    :return: The background subtracted image
    """
    normalized_radius = radius // 255
    kernel = restoration.ellipsoid_kernel(
        (radius * 2, radius * 2),
        normalized_radius * 2
    )
    rolling_ball = restoration.rolling_ball(img, radius=radius, kernel=kernel)
    return img - rolling_ball


def frangi_filter(img, sigma_range, black_ridges):
    """
    Apply the Frangi filter to an image
    :param img: The image to apply the filter to
    :param sigma_range: The range of sigmas to use (start, stop, step)
    :param black_ridges: Whether to detect black ridges
    :return: The filtered image
    """
    return frangi(img, sigmas=[x for x in range(*sigma_range)], black_ridges=black_ridges)


def li_threshold_image(img):
    """
    Apply the Li thresholding algorithm to an image
    :param img: The image to apply the thresholding to
    :return: The thresholded image
    """
    return img > thresholding.threshold_li(img, initial_guess=np.quantile(img, 0.95))


def sauvola_threshold_image(img, window_size=15):
    """
    Apply the Sauvola thresholding algorithm to an image
    :param img: The image to apply the thresholding to
    :param window_size: The size of the window to use for thresholding
    :return: The thresholded image
    """
    return img > filters.threshold_sauvola(img, window_size=window_size)


def estimate_tissue_mask(img):
    """
    Estimate the tissue mask from an image
    :param img: The image to estimate the mask from
    :return: The tissue mask
    Based on function found here: https://github.com/joe-from-mtl/sbhassisant-2d-3d-registration
    """
    mask_data = img > 0
    # Get a tissue threshold value
    threshold_tissue = thresholding.threshold_triangle(img[mask_data])

    # Apply threshold
    mask = img > threshold_tissue

    # Filter out noisy segmentation
    mask = median_filter(mask, 5)

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


def erode_mask(mask, disk_size=30):
    """
    Erode the outer edge of a mask
    :param mask: The mask to erode
    :param disk_size: The size of the disk to use for erosion
    :return: The eroded mask
    """
    return binary_erosion(mask, disk(disk_size))


def segment_2d_image(output_dir, image, name, frangi_sigma_range=(2, 16, 2), frangi_black_ridges=False,
                     local_threshold=False, local_threshold_size=15):
    """
    Segment 2D images. Finished files are not returned due to memory concerns.

    :param output_dir: The directory to save the results to
    :param image: The image to segment
    :param name: The name of the image
    :param frangi_sigma_range: The range of sigmas to use for the Frangi filter
    :param frangi_black_ridges: Whether to detect black ridges
    :param local_threshold: Whether to use local thresholding
    :param local_threshold_size: The size of the local thresholding window, must be odd
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
