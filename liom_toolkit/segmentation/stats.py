"""Per-region vessel morphometric statistics and Allen atlas region filtering."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import dask.array as da
import imageio.v3 as iio
import numpy as np
import pandas as pd
import PIL.Image
import scipy.ndimage as ndi
from dask.distributed import Future
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter
from skimage.measure import label
from skimage.measure._regionprops import RegionProperties
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from tqdm.auto import tqdm

from liom_toolkit.utils.dask_client import dask_client_manager

PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000  # finite DoS-guard limit (not None — AGENTS §2)


def compute_slice_metrics(
    output_dir: str,
    image: str,
    mask: NDArray[np.number],
    vessel_mask: NDArray[np.number],
    region_map: NDArray[np.number],
    vessel_exclude: NDArray[np.number],
    voxel_size: float = 0.65,
) -> None:
    """Compute the metrics for a brain slice and save the results to disk.

    ``image`` is a string label (filename/identifier) used in the output
    DataFrame and progress messages, not an image array. The actual image
    arrays are ``mask``, ``vessel_mask``, ``region_map``, and
    ``vessel_exclude``.

    Vessel-free regions: regions with no vessels yield a row with vessel
    density = 0.0, vessel area = 0.0, and branching points = 0, but the
    'mean diameter (um)' entry is OMITTED (the row has no such column value)
    because the mean diameter of an empty vessel set is undefined. The
    omitted diameter row is itself a publishable 'no vessels detected'
    signal, not a silent gap.

    Parameters
    ----------
    output_dir : str
        The directory to save the output to.
    image : str
        The label (filename/identifier) of the brain slice, used in the
        output DataFrame and progress messages.
    mask : ArrayLike
        The mask of the tissue in the brain slice.
    vessel_mask : ArrayLike
        The mask of the vessels in the brain slice.
    region_map : ArrayLike
        The map of the regions in the brain slice.
    vessel_exclude : ArrayLike
        The mask of the vessels to exclude from the analysis.
    voxel_size : float
        The size of the voxels in the image.

    Notes
    -----
    ``ValueError`` propagates from :func:`calculate_regional_density` when a
    region has zero area (bad region mask, caller error).
    """
    # Setup output directory (overwrite-safe via the symlink-aware
    # create_directory helper from utils.zarr_writer: a second call into an
    # existing output_dir shutil.rmtree's the directory then recreates it,
    # eliminating the FileExistsError race on re-run. Function-scope import
    # avoids a circular import with utils.zarr_writer at module load time,
    # matching the conversion.py:save_zarr pattern.)
    from liom_toolkit.utils.zarr_writer import create_directory

    create_directory(Path(output_dir), overwrite=True)
    df = pd.DataFrame(
        columns=[
            "image",
            "region",
            "vessel area (um2)",
            "tissue area (um2)",
            "vessel density (um2/um2)",
            "branching points",
            "mean diameter (um)",
        ]
    )

    # Get the different brain regions
    regions, region_count = label(region_map, return_num=True)
    props_list = measure.regionprops(regions)

    full_vessel_mask = vessel_mask * mask
    full_vessel_mask = full_vessel_mask * vessel_exclude

    # Compute metrics per region
    for i in tqdm(
        range(region_count), desc="Computing metrics per region for " + image, leave=False
    ):
        region = get_vessel_region(regions, i, full_vessel_mask)

        # Calculate vessel density
        vessel_area, total_area, density = calculate_regional_density(
            region, i, props_list, output_dir, voxel_size
        )

        # Count branching points
        branching_points_count, skeleton, branching_points = get_branching_point_count(
            region, output_dir, filename=str(i) + "_skeleton.tif"
        )
        draw_branch_point_circles(
            skeleton, branching_points, output_dir, filename=str(i) + "_skeleton_circled.png"
        )

        # Calculate average diameter. Vessel-free regions raise ValueError
        # from compute_average_diameter (mean diameter of an empty set is
        # undefined); the diameter row is OMITTED for those regions while
        # the density=0.0 row above is kept. The omitted diameter row is
        # itself a publishable 'no vessels detected' signal, not a silent
        # gap.
        try:
            mean_diameter = compute_average_diameter(region, skeleton, voxel_size)
            # Save data
            entry = pd.DataFrame.from_dict(
                {
                    "image": [image],
                    "region": [i],
                    "vessel area (um2)": [vessel_area],
                    "tissue area (um2)": [total_area],
                    "vessel density (um2/um2)": [density],
                    "branching points": [branching_points_count],
                    "mean diameter (um)": [mean_diameter],
                }
            )
        except ValueError:
            # Vessel-free region: density=0.0 row kept, diameter row omitted.
            entry = pd.DataFrame.from_dict(
                {
                    "image": [image],
                    "region": [i],
                    "vessel area (um2)": [vessel_area],
                    "tissue area (um2)": [total_area],
                    "vessel density (um2/um2)": [density],
                    "branching points": [branching_points_count],
                }
            )
        df = pd.concat([df, entry])

    # Compute metrics for the whole slice. Use full_vessel_mask
    # (vessel_mask * mask * vessel_exclude) for the density calculation so
    # the 'total' row is consistent with the per-region rows, which all
    # derive their vessel area from full_vessel_mask via get_vessel_region.
    # The pre-fix code passed the raw vessel_mask, so the total row's
    # vessel_area and vessel_density included vessels outside the tissue
    # mask and explicitly excluded vessels -- making the total density
    # higher than the sum of per-region densities (a data inconsistency
    # in the same output DataFrame).
    tissue_area, vessel_area, vessel_density = calculate_density(full_vessel_mask, mask, voxel_size)
    # Use full_vessel_mask (vessel_mask * mask * vessel_exclude) for the
    # whole-slice branching and diameter calculations so the 'total' row is
    # consistent with the per-region rows, which all derive their vessel area
    # from full_vessel_mask via get_vessel_region. The pre-fix code passed
    # the raw vessel_mask, so the total row's branching count and diameter
    # included vessels outside the tissue mask and explicitly excluded
    # vessels -- inconsistent with the per-region rows in the same DataFrame.
    branching_points_count, skeleton, branching_points = get_branching_point_count(
        full_vessel_mask, output_dir
    )
    draw_branch_point_circles(skeleton, branching_points, output_dir)
    # Whole-slice diameter: wrap in try/except ValueError mirroring the
    # per-region row-omission pattern above. A vessel-free slice (no vessels
    # anywhere) hits the empty-vessel-set ValueError from
    # compute_average_diameter (D-01 contract); without this wrap the whole
    # metrics computation crashes and the user loses every per-region row
    # computed up to this point. The 'total' row keeps the density=0.0 /
    # branching-points / vessel-area entries and OMITS the mean-diameter
    # entry, the same publishable 'no vessels detected' signal used for
    # vessel-free regions.
    try:
        mean_diameter = compute_average_diameter(full_vessel_mask, skeleton, voxel_size)
        total_entry = {
            "image": [image],
            "region": "total",
            "vessel area (um2)": [vessel_area],
            "tissue area (um2)": [tissue_area],
            "vessel density (um2/um2)": [vessel_density],
            "branching points": [branching_points_count],
            "mean diameter (um)": [mean_diameter],
        }
    except ValueError:
        # Vessel-free slice: density=0.0 row kept, diameter row omitted.
        total_entry = {
            "image": [image],
            "region": "total",
            "vessel area (um2)": [vessel_area],
            "tissue area (um2)": [tissue_area],
            "vessel density (um2/um2)": [vessel_density],
            "branching points": [branching_points_count],
        }

    # Save intermediate results. The mask/region/skeleton arrays are
    # small-integer label/mask arrays, so an explicit .astype(np.uint8)
    # narrowing is used instead of skimage.util.img_as_ubyte: img_as_ubyte
    # on an integer input whose max fits in uint8 emits a "Downcasting ...
    # without scaling" UserWarning (skimage telling us it skipped the
    # rescale), and the explicit astype is identical output with no warning
    # and clearer intent (we are narrowing a small-integer mask, not
    # rescaling a float image).
    iio.imwrite(str(Path(output_dir) / "regions.png"), regions.astype(np.uint8))
    iio.imwrite(str(Path(output_dir) / "vessel_exclude.png"), vessel_exclude.astype(np.uint8))
    iio.imwrite(str(Path(output_dir) / "_complete_mask.png"), mask.astype(np.uint8))
    iio.imwrite(str(Path(output_dir) / "vessels.png"), vessel_mask.astype(np.uint8))

    # Save data
    entry = pd.DataFrame.from_dict(total_entry)
    df = pd.concat([df, entry])
    df.to_excel(str(Path(output_dir) / "regions.xlsx"), index=False)


def get_vessel_region(
    regions: NDArray[np.number], region_index: int, vessel_mask: NDArray[np.number]
) -> NDArray[np.number]:
    """Get the vessels in a region.

    Parameters
    ----------
    regions : ArrayLike
        The regions of the tissue mask.
    region_index : int
        The index of the region.
    vessel_mask : ArrayLike
        The mask of the vessels.

    Returns
    -------
    NDArray[np.generic]
        The vessel within the masked region.
    """
    region = regions == region_index + 1
    return region * vessel_mask


def calculate_regional_density(
    region: NDArray[np.number],
    region_index: int,
    props_list: list[RegionProperties],
    output_dir: str,
    voxel_size: float = 0.65,
) -> tuple[float, float, float]:
    """Calculate the density of vessels in a region.

    Parameters
    ----------
    region : ArrayLike
        The region to calculate the density of.
    region_index : int
        The computational index of the region.
    props_list : list[RegionProperties]
        The list of properties of the regions.
    output_dir : str
        The directory to save the region mask to.
    voxel_size : float
        The size of the voxels in the image.

    Returns
    -------
    tuple[float, float, float]
        The area of the vessels, the area of the region, and the density of
        the vessels in a specific region.

    Raises
    ------
    ValueError
        If the region has zero area (bad region mask, caller error).
    """
    vessel_area = float((region == 1).sum()) * math.pow(voxel_size, 2)
    total_area = float(props_list[region_index].area) * math.pow(voxel_size, 2)
    if total_area == 0:
        raise ValueError("Empty region: regionprops area is 0 (bad region mask, caller error)")
    iio.imwrite(str(Path(output_dir) / f"{region_index}.tif"), region.astype(np.uint8))
    density = vessel_area / total_area
    return vessel_area, total_area, density


def calculate_density(
    vessel_mask: NDArray[np.number], mask: NDArray[np.number], voxel_size: float = 0.65
) -> tuple[float, float, float]:
    """Calculate the areas of the tissue and vessel to compute vessel density in a mask.

    Empty-result contract:

    * Empty vessel mask over positive tissue (``mask.sum() > 0``) returns
      ``(tissue_area, 0.0, 0.0)`` -- 0 vessels / positive tissue area is a
      well-defined 0 density (the math is defined).
    * Empty tissue mask (``mask.sum() == 0``) raises ``ValueError`` -- an
      empty tissue mask is a bad region mask and a caller error, not a
      valid result.

    Parameters
    ----------
    vessel_mask : ArrayLike
        The mask of the vessels.
    mask : ArrayLike
        The mask of the tissue.
    voxel_size : float
        The size of the voxels in the image.

    Returns
    -------
    tuple[float, float, float]
        The area of the tissue, the area of the vessels, and the density of
        the vessels.

    Raises
    ------
    ValueError
        If the tissue mask is empty (``mask.sum() == 0``).
    """
    tissue_area = float(mask.sum()) * math.pow(voxel_size, 2)
    if tissue_area == 0:
        raise ValueError("Empty tissue mask: mask.sum() == 0 (bad region mask, caller error)")
    vessel_area = float(vessel_mask.sum()) * math.pow(voxel_size, 2)
    vessel_density = vessel_area / tissue_area
    return tissue_area, vessel_area, vessel_density


def get_branching_point_count(
    vessel_mask: NDArray[np.number], output_dir: str, filename: str = "skeleton.tif"
) -> tuple[int, NDArray[np.bool_], NDArray[np.bool_]]:
    """Get the number of branching points in a vessel mask.

    Parameters
    ----------
    vessel_mask : ArrayLike
        The mask of the vessels.
    output_dir : str
        The directory to save the skeleton to.
    filename : str
        The filename to save the skeleton to.

    Returns
    -------
    tuple[int, NDArray[np.bool_], NDArray[np.bool_]]
        The number of branching points in the vessel mask, the skeleton of
        the vessel mask, and the location of the branching points.
    """
    skeleton = skeletonize(vessel_mask)
    branching_points = get_branching_points(skeleton)
    points_count = branching_points.sum()
    iio.imwrite(str(Path(output_dir) / filename), skeleton.astype(np.uint8))
    return points_count, skeleton, branching_points


def get_branching_points(skeleton: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Get the branching points in a skeleton using predefined structural elements.

    Source:
    https://stackoverflow.com/questions/43037692/how-to-find-branch-point-from-binary-skeletonize-image

    Parameters
    ----------
    skeleton : ArrayLike
        The skeleton of the vessels.

    Returns
    -------
    NDArray[np.bool_]
        The branching points in the skeleton.
    """
    # Setup structural elements for detecting branching points
    selems = []
    selems.extend(
        (
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]),
        )
    )
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    # Detect branching points
    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)
    return branches


def draw_branch_point_circles(
    skeleton: NDArray[np.bool_],
    branching_points: NDArray[np.bool_],
    output_dir: str,
    filename: str = "skeleton_circled.png",
) -> None:
    """Draw circles around the branching points in a skeleton and save to disk.

    Parameters
    ----------
    skeleton : ArrayLike
        The skeleton of the vessels.
    branching_points : ArrayLike
        The location of the branching points.
    output_dir : str
        The directory to save the skeleton to.
    filename : str
        The filename to save the skeleton to.
    """
    circled_skeleton = gray2rgb(skeleton.astype(np.uint8))
    points_to_draw = np.argwhere(branching_points)
    for point in points_to_draw:
        circy, circx = circle_perimeter(point[0], point[1], 7, shape=skeleton.shape)
        circled_skeleton[circy, circx] = (220, 20, 20)

    iio.imwrite(str(Path(output_dir) / filename), circled_skeleton)
    del circled_skeleton


def compute_average_diameter(
    mask: NDArray[np.number], skeleton: NDArray[np.bool_], voxel_size: float = 0.65
) -> float:
    """Compute the average diameter of the vessels in a mask.

    Empty-result contract:

    * Empty vessel set (no positive radii in the skeleton) raises
      ``ValueError`` -- the mean diameter of an empty set is undefined;
      returning 0.0 would imply zero-width vessels exist (a
      plausible-shaped-but-wrong value), and returning ``NaN`` would let a
      silent NaN escape into the published pandas DataFrame. The raise sits
      BEFORE ``np.mean`` so no NaN + RuntimeWarning can escape.
    * Empty tissue mask raises ``ValueError`` -- mean diameter is undefined
      when there is no tissue.

    Parameters
    ----------
    mask : ArrayLike
        The vessel mask.
    skeleton : ArrayLike
        The skeleton of the vessels.
    voxel_size : float
        The size of the voxels in the image.

    Returns
    -------
    float
        The average diameter of the vessels in the mask.

    Raises
    ------
    ValueError
        If the vessel mask is empty or no positive radii are found in the
        skeleton.
    TypeError
        If the computed average diameter is not a float.
    """
    if mask.sum() == 0:
        raise ValueError("Empty tissue mask: mean diameter is undefined")
    distance = distance_transform_edt(mask.astype(np.float64))
    radii = distance * skeleton.astype(bool)
    positive_radii = radii[radii > 0]
    if positive_radii.size == 0:
        raise ValueError(
            "Empty vessel set: mean diameter is undefined (no positive radii in skeleton)"
        )
    mean_radius = np.mean(positive_radii)
    mean_diameter = 2 * mean_radius
    result = mean_diameter * voxel_size
    if not isinstance(result, float):
        raise TypeError(f"Expected float, got {type(result)}")
    return result


def create_heatmap(image: NDArray[np.number], output_dir: str, square_size: int = 150) -> None:
    """Create and save a heatmap of the vessel density in a brain slice.

    Parameters
    ----------
    image : ArrayLike
        The image of the brain slice.
    output_dir : str
        The directory to save the heatmap to.
    square_size : int
        The size of the squares in the heatmap.
    """
    # Overwrite-safe output directory creation (same create_directory helper
    # as compute_slice_metrics above; function-scope import avoids a circular
    # import with utils.zarr_writer at module load time).
    from liom_toolkit.utils.zarr_writer import create_directory

    create_directory(Path(output_dir), overwrite=True)

    image = img_as_ubyte(image)
    image = image / 255
    image = image.astype(np.uint8)
    heatmap = np.zeros_like(image, dtype=np.uint32)
    # Compute the per-dimension block counts separately so non-square images
    # iterate the correct number of squares per dimension (a single
    # shape[0]/square_size count for both dims produces a staircase pattern
    # on non-square or multi-row images).
    n_x = int(image.shape[0] / square_size)
    n_y = int(image.shape[1] / square_size)
    x_start = 0
    for _ in range(n_x):
        y_start = 0  # reset at the start of each outer iteration
        for _j in range(n_y):
            heatmap[x_start : x_start + square_size, y_start : y_start + square_size] = image[
                x_start : x_start + square_size, y_start : y_start + square_size
            ].sum()
            y_start += square_size
        x_start += square_size

    # Set final square to max value to ensure same scaling across heatmaps
    heatmap[-1, -1] = square_size**2
    # heatmap is uint32 with max square_size**2 (e.g. 22500), which fits in
    # uint16 without scaling. Use an explicit .astype(np.uint16) narrowing
    # instead of skimage.util.img_as_uint: img_as_uint on an integer input
    # whose max fits in uint16 emits a "Downcasting ... without scaling"
    # UserWarning, and the explicit astype is identical output with no
    # warning and clearer intent.
    heatmap = heatmap.astype(np.uint16)
    heatmap = heatmap.astype(float)
    heatmap = heatmap / (square_size**2)
    iio.imwrite(str(Path(output_dir) / "heatmap.tif"), heatmap)


def generate_itk_id_list_of_region(region: str, data_dir: str = "") -> list[int]:
    """Generate a list of itk ids for a given region.

    Reconstructs the structure tree and gets the descendants contained
    within the region.

    Parameters
    ----------
    region : str
        The region to get the ids for.
    data_dir : str
        The directory where the atlas and structure tree are saved. Optional.

    Returns
    -------
    list[int]
        The list of itk ids for the region and its descendants.

    Raises
    ------
    TypeError
        If the extracted itk ids are not a list.
    """
    # Setup temporary directory if not given. Track whether WE created it
    # so the cleanup runs unconditionally (the pre-fix code reassigned
    # data_dir = temp_dir.name then re-tested ``if data_dir == ""``, which
    # was always False after the reassignment, so temp_dir.cleanup() never
    # ran and the temp directory leaked on every call where data_dir="" --
    # the default).
    use_temp = data_dir == ""
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if use_temp:
        temp_dir = tempfile.TemporaryDirectory()
        data_dir = temp_dir.name

    itk_ids: list[Any] = []
    try:
        # Construct reference space and get itk ids
        from liom_toolkit.utils import construct_reference_space

        rs = construct_reference_space(data_dir)
        structure_tree = rs.structure_tree
        _, labels = rs.export_itksnap_labels()

        # Get the itk ids for the region
        region_structures = structure_tree.get_structures_by_name([region])
        region_id = region_structures[0]["id"]
        region_sub = structure_tree.descendant_ids([region_id])
        region_sub_acronyms = [
            region["acronym"] for region in structure_tree.get_structures_by_id(region_sub[0])
        ]
        itk_ids = labels.loc[labels["LABEL"].isin(region_sub_acronyms)]["IDX"].to_numpy().tolist()
    finally:
        if use_temp and temp_dir is not None:
            temp_dir.cleanup()

    if not isinstance(itk_ids, list):
        raise TypeError(f"Expected list, got {type(itk_ids)}")
    return [int(x) for x in itk_ids]


def create_filter_image(atlas: da.Array | Future[Any], region_ids: list[int]) -> da.Array:
    """Create a filter image based on the region ids.

    Parameters
    ----------
    atlas : da.Array | Future[Any]
        The atlas containing the region ids.
    region_ids : list[int]
        The region ids to filter.

    Returns
    -------
    da.Array
        The filter image.

    Raises
    ------
    TypeError
        If the gathered filter image is not a Dask array.
    """
    client = dask_client_manager.get_client()
    filter_image = client.submit(da.isin, atlas, region_ids)
    result = client.gather(filter_image)
    if not isinstance(result, da.Array):
        raise TypeError(f"Expected dask Array, got {type(result)}")
    return result


def filter_image_to_region(image_filter: da.Array, data: da.Array | Future[Any]) -> da.Array:
    """Filter an image to a region based on a filter.

    Parameters
    ----------
    image_filter : da.Array
        The filter to apply.
    data : da.Array | Future[Any]
        The data to filter.

    Returns
    -------
    da.Array
        The filtered image.

    Raises
    ------
    TypeError
        If the gathered filtered image is not a Dask array.
    """
    client = dask_client_manager.get_client()
    filtered_image = client.submit(da.where, image_filter, data, 0)
    result = client.gather(filtered_image)
    if not isinstance(result, da.Array):
        raise TypeError(f"Expected dask Array, got {type(result)}")
    return result


def compute_mask_area(mask: da.Array | Future[Any]) -> np.uint64:
    """Compute the area of a mask by summing the binary mask values.

    Parameters
    ----------
    mask : da.Array | Future[Any]
        The mask to compute the area of.

    Returns
    -------
    np.uint64
        The area of the mask.
    """
    client = dask_client_manager.get_client()
    total_area = client.submit(da.sum, mask)
    total_area = client.gather(total_area)
    result = total_area.compute()
    return np.uint64(result)
