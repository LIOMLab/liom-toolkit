import numpy as np
from scipy.ndimage import median_filter, binary_fill_holes
from skimage import restoration, img_as_ubyte, filters
from skimage.filters import frangi, thresholding
from skimage.io import imread, imsave
from skimage.measure import regionprops, label
from skimage.morphology import disk, binary_erosion
from tqdm import tqdm


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


def sauvola_threshold_image(img):
    """
    Apply the Sauvola thresholding algorithm to an image
    :param img: The image to apply the thresholding to
    :return: The thresholded image
    """
    return img > filters.threshold_sauvola(img)


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

    # Fill holes
    mask = binary_fill_holes(mask)

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


def segment_2d_images(base_directory, images, erode_mask_size=30, background_filter_size=70,
                      frangi_sigma_range=(2, 16, 2), frangi_black_ridges=False, local_threshold=False):
    """
    Segment 2D images
    :param base_directory: The base directory to save the results to and load the images from
    :param images: The filenames of the images to segment
    :param erode_mask_size: The size of the disk to use for erosion
    :param background_filter_size: The size of the background filter
    :param frangi_sigma_range: The range of sigmas to use for the Frangi filter (start, stop, step)
    :param frangi_black_ridges: Whether to detect black ridges
    :param local_threshold: Wheter to apply local or global thresholding
    :return Lists of segmented images, both the Frangi filter and the thersholded vessel mask
    """
    vessel_masks = []
    frangi_images = []
    pbar = tqdm(images)
    for image in pbar:
        # Read image
        img = imread(base_directory + images[0], as_gray=True)

        pbar.set_description("Creating mask")
        # Create full mask
        mask = estimate_tissue_mask(img)

        pbar.set_description("Eroding mask")
        # Erode outer edge
        erode = erode_mask(mask, disk_size=erode_mask_size)

        pbar.set_description("Removing background")
        # Remove background from image
        bg_less = subtract_background(img, background_filter_size)

        pbar.set_description("Applying Frangi filter")
        # Apply Frangi filter
        frangi = frangi_filter(bg_less, frangi_sigma_range, frangi_black_ridges)

        pbar.set_description("Applying threshold")
        # Apply threshold
        if local_threshold:
            vessel_mask_raw = sauvola_threshold_image(frangi)
        else:
            vessel_mask_raw = li_threshold_image(frangi)

        pbar.set_description("Apply erosion to mask")
        # Apply erosion
        vessel_mask = vessel_mask_raw * erode

        pbar.set_description("Saving images")
        # Save image
        imsave(base_directory + image + '_vessel_mask.tif', img_as_ubyte(vessel_mask))
        imsave(base_directory + image + '_frangi.tif', frangi, check_contrast=False)
        vessel_masks.append(vessel_mask)
        frangi_images.append(frangi)
    return vessel_masks, frangi_images
