import os

import numpy as np

from scipy.ndimage import median_filter, binary_fill_holes
from skimage.morphology import disk, binary_erosion
from skimage.filters import frangi, thresholding
from skimage import restoration
from skimage.measure import regionprops, label
from skimage.io import imread, imsave
from tqdm import tqdm

home_directory = os.path.expanduser('~')


def subtract_background(img, radius=70):
    normalized_radius = radius // 255
    kernel = restoration.ellipsoid_kernel(
        (radius * 2, radius * 2),
        normalized_radius * 2
    )
    rolling_ball = restoration.rolling_ball(img, radius=radius, kernel=kernel)
    return img - rolling_ball


def remove_background(img, radius=70):
    return subtract_background(img, radius=radius)


def frangi_filter(img, sigma_range, black_ridges):
    return frangi(img, sigmas=[x for x in range(sigma_range)], black_ridges=black_ridges)


def li_threshold_image(img):
    return img > thresholding.threshold_li(img, initial_guess=np.quantile(img, 0.95))


def local_threshold_image(img):
    return img > thresholding.threshold_local(img, block_size=501)


def create_vessel_mask(img, background_filter_size=70, frangi_sigma_range=(2, 16, 2), frangi_black_ridges=False):
    bg_less = remove_background(img, background_filter_size)
    frangi_image = frangi_filter(bg_less, frangi_sigma_range, frangi_black_ridges)
    thresholded_image = li_threshold_image(frangi_image)
    return thresholded_image, frangi_image


def estimate_tissue_mask(img):
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
    return binary_erosion(mask, disk(disk_size))


def segment_2d_images(base_directory, images, erode_mask_size=30, background_filter_size=70,
                      frangi_sigma_range=(2, 16, 2), frangi_black_ridges=False):
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

        pbar.set_description("Creating vessel mask")
        # Create Vessel mask
        vessel_mask_raw, frangi = create_vessel_mask(img, background_filter_size, frangi_sigma_range,
                                                     frangi_black_ridges)

        pbar.set_description("Apply erosion to mask")
        # Apply erosion
        vessel_mask = vessel_mask_raw * erode

        pbar.set_description("Saving images")
        # Save image
        imsave(base_directory + image + '_vessel_mask.tif', vessel_mask)
        imsave(base_directory + image + '_frangi.tif', frangi)
