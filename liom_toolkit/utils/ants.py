"""Dask/zarr to ANTsImage bridge (ants is lazy-imported)."""

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
    volume_direction: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
    ),
) -> ANTsImage:
    """Convert a dask array to an ANTs image.

    Parameters
    ----------
    dask_array : da.Array
        The dask array to convert.
    node : Node
        The zarr node corresponding to the image.
    resolution_level : int
        The resolution level to load.
    volume_direction : tuple[tuple[float, float, float], ...]
        The direction of the volume (3x3 row-major direction matrix).

    Returns
    -------
    ANTsImage
        The converted ANTs image.

    Raises
    ------
    ImportError
        If ``ants`` (ANTsPy) is not installed.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the ants utility functions of the LIOM toolkit."
        ) from e
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


def load_ants_image_from_node(node: Node, resolution_level: int = 2, channel: int = 0) -> ANTsImage:
    """Load an ANTs image from a zarr node.

    Parameters
    ----------
    node : Node
        The zarr node to load.
    resolution_level : int
        The resolution level to load.
    channel : int
        The channel to load.

    Returns
    -------
    ANTsImage
        The loaded ANTs image.

    Raises
    ------
    ImportError
        If ``ants`` (ANTsPy) is not installed.
    """
    try:
        import ants  # ruff: ignore[unused-import] -- imported for the actionable error; convert_dask_to_ants re-imports
    except ImportError as e:
        raise ImportError(
            "Please install ANTsPy to use the ants utility functions of the LIOM toolkit."
        ) from e
    image = load_zarr_image_from_node(node, resolution_level)
    if len(image.shape) == 4:
        image = image[channel, :, :, :]
    return convert_dask_to_ants(image, node, resolution_level)
