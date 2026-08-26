from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from ome_zarr.reader import Node

from .io import load_zarr_image_from_node, load_zarr_transform_from_node

if TYPE_CHECKING:
    from ants import ANTsImage


def convert_dask_to_ants(
    dask_array: da.Array,
    node: Node,
    resolution_level: int = 2,
    volume_direction: tuple = ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
) -> ANTsImage:
    """
    Convert a dask array to an ANTs image.

    :param dask_array: The dask array to convert.
    :type dask_array: da.Array
    :param node: The zarr node corresponding to the image.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :param volume_direction: The direction of the volume.
    :type volume_direction: tuple
    :return: The converted ANTs image.
    :rtype: ANTsImage
    """
    try:
        import ants
    except ImportError:
        raise ImportError(
            "Please install ANTsPy to use the ants utility functions of the LIOM toolkit."
        )
    # Compute dask array to get values
    array = dask_array.compute()

    # reverse the order of the axes
    array = np.transpose(array, (2, 1, 0)).astype("uint32")
    ants_image = ants.from_numpy(array)

    # Set metadata
    transform = load_zarr_transform_from_node(node, resolution_level=resolution_level)
    if len(transform) == 4:
        transform = transform[1:]

    # Convert to mm
    transform = [element / 1000 for element in transform]
    ants_image.set_spacing(transform)
    ants_image.set_direction(volume_direction)

    return ants_image


def load_ants_image_from_node(node: Node, resolution_level: int = 2, channel=0) -> ANTsImage:
    """
    Load an ANTs image from a zarr node.

    :param node: The zarr node to load.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :param channel: The channel to load.
    :type channel: int
    :return: The loaded ANTs image.
    :rtype: ANTsImage
    """
    try:
        import ants  # noqa: F401 -- imported for the actionable error; convert_dask_to_ants re-imports
    except ImportError:
        raise ImportError(
            "Please install ANTsPy to use the ants utility functions of the LIOM toolkit."
        )
    image = load_zarr_image_from_node(node, resolution_level)
    if len(image.shape) == 4:
        image = image[channel, :, :, :]
    ants_image = convert_dask_to_ants(image, node, resolution_level)
    return ants_image
