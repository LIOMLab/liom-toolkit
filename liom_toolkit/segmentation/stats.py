import math
import os

import PIL.Image
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.io import imsave
from skimage.measure import label
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte, img_as_uint
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = None


def compute_slice_metrics(output_dir: str, image: np.ndarray, mask: np.ndarray, vessel_mask: np.ndarray,
                          region_map: np.ndarray, vessel_exclude: np.ndarray, voxel_size: float = 0.65) -> None:
    """
    Compute the metrics for a brain slice. Save the results to disk.

    :param output_dir: The directory to save the output to
    :type output_dir: str
    :param image: The image of the brain slice
    :type image: np.ndarray
    :param mask: The mask of the tissue in the brain slice
    :type mask: np.ndarray
    :param vessel_mask: The mask of the vessels in the brain slice
    :type vessel_mask: np.ndarray
    :param region_map: The map of the regions in the brain slice
    :type region_map: np.ndarray
    :param vessel_exclude: The mask of the vessels to exclude from the analysis
    :type vessel_exclude: np.ndarray
    :param voxel_size: The size of the voxels in the image
    :type voxel_size: float
    """

    # Setup output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df = pd.DataFrame(columns=['image', 'region', 'vessel area (um2)', 'tissue area (um2)', 'vessel density (um2/um2)',
                               'branching points', 'mean diameter (um)'])

    # Get the different brain regions
    regions, region_count = label(region_map, return_num=True)
    props_list = measure.regionprops(regions)

    full_vessel_mask = vessel_mask * mask
    full_vessel_mask = full_vessel_mask * vessel_exclude

    # Compute metrics per region
    for i in tqdm(range(0, region_count), desc="Computing metrics per region for " + image, leave=False):
        region = get_vessel_region(regions, i, full_vessel_mask)

        # Calculate vessel density
        vessel_area, total_area, density = calculate_regional_density(region, i, props_list, output_dir, voxel_size)

        # Count branching points
        branching_points_count, skeleton, branching_points = get_branching_point_count(region, output_dir, filename=str(
            i) + '_skeleton.tif')
        draw_branch_point_circles(skeleton, branching_points, output_dir, filename=str(i) + '_skeleton_circled.png')

        # Calculate average diameter
        mean_diameter = compute_average_diameter(region, skeleton, voxel_size)

        # Save data
        entry = pd.DataFrame.from_dict({'image': [image], 'region': [i], 'vessel area (um2)': [vessel_area],
                                        'tissue area (um2)': [total_area], 'vessel density (um2/um2)': [density],
                                        'branching points': [branching_points_count],
                                        'mean diameter (um)': [mean_diameter]})
        df = pd.concat([df, entry])

    # Compute metrics for the whole slice
    tissue_area, vessel_area, vessel_density = calculate_density(vessel_mask, mask, voxel_size)
    branching_points_count, skeleton, branching_points = get_branching_point_count(vessel_mask, output_dir)
    draw_branch_point_circles(skeleton, branching_points, output_dir)
    mean_diameter = compute_average_diameter(vessel_mask, skeleton, voxel_size)

    # Save intermediate results
    imsave(output_dir + 'regions.png', img_as_ubyte(regions), check_contrast=False)
    imsave(output_dir + 'vessel_exclude.png', img_as_ubyte(vessel_exclude), check_contrast=False)
    imsave(output_dir + '_complete_mask.png', img_as_ubyte(mask), check_contrast=False)
    imsave(output_dir + 'vessels.png', img_as_ubyte(vessel_mask), check_contrast=False)

    # Save data
    entry = pd.DataFrame.from_dict({'image': [image], 'region': 'total', 'vessel area (um2)': [vessel_area],
                                    'tissue area (um2)': [tissue_area], 'vessel density (um2/um2)': [vessel_density],
                                    'branching points': [branching_points_count],
                                    'mean diameter (um)': [mean_diameter]})
    df = pd.concat([df, entry])
    df.to_excel(output_dir + 'regions.xlsx', index=False)


def get_vessel_region(regions: np.ndarray, region_index: int, vessel_mask: np.ndarray) -> np.ndarray:
    """
    Get the vessels in a region.

    :param regions: The regions of the tissue mask.
    :type regions: np.ndarray
    :param region_index: The index of the region.
    :type region_index: int
    :param vessel_mask: The mask of the vessels.
    :type vessel_mask: np.ndarray
    :return: The vessel within the masked region.
    :rtype: np.ndarray
    """
    region = regions == region_index + 1
    region = region * vessel_mask
    return region


def calculate_regional_density(region: np.ndarray, region_index: int, props_list: list[RegionProperties],
                               output_dir: str, voxel_size: float = 0.65) -> tuple[float, float, float]:
    """
    Calculates the density of vessels in a region

    :param region: The region to calculate the density of.
    :type region: np.ndarray
    :param region_index: The computational index of the region.
    :type region_index: int
    :param props_list: The list of properties of the regions.
    :type props_list: list[RegionProperties]
    :param output_dir: The directory to save the region mask to.
    :type output_dir: str
    :param voxel_size: The size of the voxels in the image.
    :type voxel_size: float
    :return: The area of the vessels, the area of the region, and the density of the vessels in a specific region.
    :rtype: tuple[float, float, float]
    """
    vessel_area = (region == 1).sum() * math.pow(voxel_size, 2)
    total_area = props_list[region_index].area * math.pow(voxel_size, 2)
    imsave(output_dir + str(region_index) + '.tif', img_as_ubyte(region), check_contrast=False)
    density = vessel_area / total_area
    return vessel_area, total_area, density


def calculate_density(vessel_mask: np.ndarray, mask: np.ndarray, voxel_size: float = 0.65) -> tuple[
    float, float, float]:
    """
    Calculates the areas of the tissue and vessel to compute the density of vessels in a mask.

    :param vessel_mask: The mask of the vessels.
    :type vessel_mask: np.ndarray
    :param mask: The mask of the tissue.
    :type mask: np.ndarray
    :param voxel_size: The size of the voxels in the image.
    :type voxel_size: float
    :return: The area of the tissue, the area of the vessels, and the density of the vessels.
    :rtype: tuple[float, float, float]
    """
    tissue_area = mask.sum() * math.pow(voxel_size, 2)
    vessel_area = vessel_mask.sum() * math.pow(voxel_size, 2)
    vessel_density = vessel_area / tissue_area
    return tissue_area, vessel_area, vessel_density


def get_branching_point_count(vessel_mask: np.ndarray, output_dir: str, filename: str = 'skeleton.tif') -> tuple[
    int, np.ndarray, np.ndarray]:
    """
    Get the number of branching points in a vessel mask.

    :param vessel_mask: The mask of the vessels.
    :type vessel_mask: np.ndarray
    :param output_dir: The directory to save the skeleton to.
    :type output_dir: str
    :param filename: The filename to save the skeleton to.
    :type filename: str
    :return: The number of branching points in the vessel mask, the skeleton of the vessel mask,
            and the location of the branching points.
    :rtype: tuple[int, np.ndarray, np.ndarray]
    """
    skeleton = skeletonize(vessel_mask)
    branching_points = get_branching_points(skeleton)
    points_count = branching_points.sum()
    imsave(output_dir + filename, img_as_ubyte(skeleton), check_contrast=False)
    return points_count, skeleton, branching_points


def get_branching_points(skeleton: np.ndarray) -> np.ndarray:
    """
    Get the branching points in a skeleton using predefined structural elements
    Source: https://stackoverflow.com/questions/43037692/how-to-find-branch-point-from-binary-skeletonize-image

    :param skeleton: The skeleton of the vessels.
    :type skeleton: np.ndarray
    :return: The branching points in the skeleton.
    :rtype: np.ndarray
    """
    # Setup structural elements for detecting branching points
    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    # Detect branching points
    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)
    return branches


def draw_branch_point_circles(skeleton: np.ndarray, branching_points: np.ndarray, output_dir: str,
                              filename: str = 'skeleton_circled.png') -> None:
    """
    Draw circles around the branching points in a skeleton

    :param skeleton: The skeleton of the vessels.
    :type skeleton: np.ndarray
    :param branching_points: The location of the branching points.
    :type branching_points: np.ndarray
    :param output_dir: The directory to save the skeleton to.
    :type output_dir: str
    :param filename: The filename to save the skeleton to.
    :type filename: str
    :return: The circled branching point in the skeleton.
    :rtype: np.ndarray
    """
    circled_skeleton = gray2rgb(img_as_ubyte(skeleton))
    points_to_draw = np.argwhere(branching_points)
    for point in points_to_draw:
        circy, circx = circle_perimeter(point[0], point[1], 7,
                                        shape=skeleton.shape)
        circled_skeleton[circy, circx] = (220, 20, 20)

    imsave(output_dir + filename, circled_skeleton, check_contrast=False)
    del circled_skeleton


def compute_average_diameter(mask: np.ndarray, skeleton: np.ndarray, voxel_size: float = 0.65) -> float:
    """
    Compute the average diameter of the vessels in a mask

    :param mask: The vessel mask.
    :type mask: np.ndarray
    :param skeleton: The skeleton of the vessels.
    :type skeleton: np.ndarray
    :param voxel_size: The size of the voxels in the image.
    :type voxel_size: float
    :return: The average diameter of the vessels in the mask.
    :rtype: float
    """
    distance = distance_transform_edt(mask)
    radii = distance * skeleton.astype(bool)
    mean_radius = np.mean(radii[radii > 0])
    mean_diameter = 2 * mean_radius
    return mean_diameter * voxel_size


def create_heatmap(image: np.ndarray, output_dir: str, square_size: int = 150) -> None:
    """
    Create and save a heatmap of the vessel density in a brain slice and save it to disk.

    :param image: The image of the brain slice.
    :type image: np.ndarray
    :param output_dir: The directory to save the heatmap to.
    :type output_dir: str
    :param square_size: The size of the squares in the heatmap
    :type square_size: int
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image = img_as_ubyte(image)
    image = image / 255
    image = image.astype(np.uint8)
    heatmap = np.zeros_like(image, dtype=np.uint32)
    x_start = 0
    y_start = 0
    for _ in range(0, int(image.shape[0] / square_size)):
        for j in range(0, int(image.shape[1] / square_size)):
            heatmap[x_start:x_start + square_size, y_start:y_start + square_size] = image[x_start:x_start + square_size,
                                                                                    y_start:y_start + square_size].sum()
            y_start += square_size
        x_start += square_size
        y_start = 0

    # Set final square to max value to ensure same scaling across heatmaps
    heatmap[-1, -1] = (square_size ** 2)
    heatmap = img_as_uint(heatmap)
    heatmap = heatmap.astype(float)
    heatmap = heatmap / (square_size ** 2)
    imsave(output_dir + 'heatmap.tif', heatmap, check_contrast=False)
