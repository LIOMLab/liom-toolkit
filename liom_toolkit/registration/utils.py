import os

import ants
import numpy as np
import pandas as pd
from ants.core.ants_image import ANTsImage
from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.reference_space_cache import ReferenceSpaceCache


def convert_allen_nrrd_to_ants(volume: np.ndarray, resolution: float) -> ANTsImage:
    """
    Convert a nrrd file form the Allen reference spaces to an ants image. The returned image will be in RAS+ orientation.

    :param volume: The already loaded nrrd file.
    :type volume: np.ndarray
    :param resolution: The resolution of the nrrd file in millimeters.
    :type resolution: float
    :return: The converted image.
    :rtype: ANTsImage
    """
    # Set axis to RAS
    volume = np.moveaxis(volume, [0, 1, 2], [1, 2, 0])

    # Convert to ants image and set direction and spacing
    volume = ants.from_numpy(volume.astype("uint32"))
    volume.set_direction([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    volume.set_spacing([resolution, resolution, resolution])

    return volume


def construct_reference_space_cache(resolution: int = 25,
                                    reference_space_key: str = "annotation/ccf_2017") -> ReferenceSpaceCache:
    """
    Construct a reference space cache for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param reference_space_key: The reference space key to use.
    :type reference_space_key: str
    :return: The reference space cache.
    :rtype: ReferenceSpaceCache
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = ReferenceSpaceCache(resolution=resolution, reference_space_key=reference_space_key)

    return rsc


def construct_reference_space(data_dir: str, resolution: int = 25,
                              reference_space_key: str = "annotation/ccf_2017") -> ReferenceSpace:
    """
    Construct a reference space for the Allen brain atlas. Will use the 2017 adult version of the atlas.

    :param data_dir: The directory where the atlas and structure tree are saved.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param reference_space_key: The reference space key to use.
    :type reference_space_key: str
    :return: The reference space.
    :rtype: ReferenceSpace
    """
    # Check the resolution
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"

    # Construct the reference space cache
    rsc = construct_reference_space_cache(resolution=resolution, reference_space_key=reference_space_key)

    # Construct the reference space
    annotation, meta = rsc.get_annotation_volume(f"{data_dir}/allen_atlas_{resolution}.nrrd")
    structure_tree = rsc.get_structure_tree(f"{data_dir}/structure_tree_{resolution}.json")
    rs = ReferenceSpace(resolution=resolution, annotation=annotation, structure_tree=structure_tree)

    return rs


def download_allen_template(data_dir: str, resolution: int = 25, keep_nrrd: bool = False,
                            rsc: ReferenceSpaceCache = None) -> ANTsImage:
    """
    Download the allen mouse brain template in RAS+ orientation.

    :param data_dir: The directory to save the template to.
    :type data_dir: str
    :param resolution: The template resolution in micron. Must be 10, 25, 50 or 100 microns
    :type resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :type keep_nrrd: bool
    :param rsc: The reference space cache to use. If None, a new one will be constructed.
    :type rsc: ReferenceSpaceCache
    :return: The template as an ants image.
    :rtype: ANTsImage
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


def download_allen_atlas(data_dir: str, resolution: int = 25, keep_nrrd: bool = False) -> (
        ANTsImage, pd.DataFrame):
    """
    Download the allen mouse brain atlas and reorient it to RAS+.

    :param data_dir: The directory to save the atlas to.
    :type data_dir: str
    :param resolution: The resolution of the atlas in micron. Must be 10, 25, 50 or 100 microns
    :param resolution: int
    :param keep_nrrd: Whether to keep the nrrd file or not.
    :param keep_nrrd: bool
    :return: The atlas as an ants image.
    :rtype:(ANTsImage, pd.DataFrame)
    """
    assert resolution in [10, 25, 50, 100], "Resolution must be 10, 25, 50 or 100"
    # Download resolution is 10 micron to fix wrong region labels

    # Temporary filename
    nrrd_file = f"{data_dir}/allen_atlas_{resolution}.nrrd"

    # Downloading the atlas
    rs = construct_reference_space(data_dir, resolution=resolution)
    vol, metadata = rs.export_itksnap_labels()

    # Convert to ants image
    ants_image = convert_allen_nrrd_to_ants(vol, resolution / 1000)

    # Remove nrrd file if unwanted
    if not keep_nrrd:
        os.remove(nrrd_file)

    return ants_image, metadata
