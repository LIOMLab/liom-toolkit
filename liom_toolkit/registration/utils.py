import os

import ants
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache


def convert_allen_nrrd_to_ants(volume: np.ndarray, resolution: float) -> ants.ANTsImage:
    """
    Convert a nrrd file form the Allen reference spaces to an ants image. The returned image will be in RAS+ orientation.

    :param volume: np.ndarray The already loaded nrrd file.
    :param resolution: float The resolution of the nrrd file in millimeters.
    :return ants.ANTsImage The converted image.
    """
    # Set axis to RAS
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    # Convert to ants image and set direction and spacing
    volume = ants.from_numpy(volume.astype("uint32"))
    volume.set_direction([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    volume.set_spacing([resolution, resolution, resolution])

    return volume


def construct_reference_space_cache(resolution: int = 25) -> ReferenceSpaceCache:
    """
    Construct a reference space cache for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param resolution: int The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :return: ReferenceSpaceCache The reference space cache.
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = ReferenceSpaceCache(resolution=resolution, reference_space_key="annotation/ccf_2017")

    return rsc


def download_allen_template(data_dir: str, resolution: int = 25, keep_nrrd: bool = False,
                            rsc: ReferenceSpaceCache = None) -> ants.ANTsImage:
    """
    Download the allen mouse brain template in RAS+ orientation.

    :param data_dir: str The directory to save the template to.
    :param resolution: int The template resolution in micron. Must be 10, 25, 50 or 100 microns
    :param keep_nrrd: bool Whether to keep the nrrd file or not.
    :param rsc: ReferenceSpaceCache The reference space cache to use. If None, a new one will be constructed.
    :return: ants.ANTsImage The template as an ants image.
    """
    # Check the resolution
    assert int(resolution) in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # filename
    nrrd_file = f"{data_dir}/allen_template_{resolution}.nrrd"

    # Downloading the template
    if rsc is None:
        rsc = construct_reference_space_cache(resolution=resolution)
    vol, metadata = rsc.get_template_volume(file_name=str(nrrd_file))

    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image


def download_allen_atlas(data_dir: str, resolution: int = 25, keep_nrrd: bool = False,
                         rsc: ReferenceSpaceCache = None) -> ants.ANTsImage:
    """
    Download the allen mouse brain atlas and reorient it to RAS+.

    :param data_dir: str The directory to save the atlas to.
    :param resolution: int The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :param keep_nrrd: bool Whether to keep the nrrd file or not.
    :param rsc: ReferenceSpaceCache The reference space cache to use. If None, a new one will be constructed.
    :return: ants.ANTsImage The atlas as an ants image.
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Temporary filename
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"

    # Downloading the annotation (if not already downloaded)
    if rsc is None:
        rsc = construct_reference_space_cache(resolution=resolution)
    vol, metadata = rsc.get_annotation_volume(str(nrrd_file))

    # Convert to ants image
    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image
