"""Slice and MIP extraction from OME-Zarr volumes.

This module provides helpers to extract single 2D slices or stacks of
slices from multiscale OME-Zarr volumes (via :class:`ome_zarr.reader.Node`)
and optionally write them to disk as PNG / TIFF, plus a label-colouring
utility for visualising segmentation masks.
"""

from __future__ import annotations

from collections.abc import Sequence

import imageio.v3 as iio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ome_zarr.reader import Node

from liom_toolkit.utils import convert_to_png_for_saving


def extract_single_slice_from_zarr(
    node: Node, z: int, channel: int = 0, resolution_level: int = 0
) -> NDArray[np.generic]:
    """Extract a single 2D slice from a 3D OME-Zarr volume.

    Parameters
    ----------
    node : Node
        The OME-Zarr node to extract the slice from.
    z : int
        Z index of the slice.
    channel : int
        Channel to extract (used when the volume is 4D).
    resolution_level : int
        Resolution level to extract from the multiscale pyramid.

    Returns
    -------
    NDArray[np.generic]
        The extracted 2D slice. The dtype matches the underlying zarr
        array (the dask array is materialised via ``.compute()``).
    """
    volume = node.data[resolution_level]
    if volume.ndim == 4:
        volume = volume[channel]

    image = volume[z, :, :]
    return image.compute()


def extract_and_save_slice_from_zarr(
    node: Node,
    z: int,
    data_dir: str,
    channel: int = 0,
    resolution_level: int = 0,
    name: str = "S1",
) -> NDArray[np.uint8]:
    """Extract a single 2D slice from a 3D OME-Zarr volume and save it to disk.

    The slice is normalised to uint8 (via :func:`convert_to_png_for_saving`)
    and written as a PNG named ``{name}_C={channel}_Z={z}.png`` under
    ``data_dir``.

    Parameters
    ----------
    node : Node
        The OME-Zarr node to extract the slice from.
    z : int
        Z index of the slice.
    data_dir : str
        Directory to save the slice PNG into.
    channel : int
        Channel to extract (used when the volume is 4D).
    resolution_level : int
        Resolution level to extract from the multiscale pyramid.
    name : str
        Name prefix of the volume, used in the output filename.

    Returns
    -------
    NDArray[np.uint8]
        The normalised uint8 slice that was written to disk.
    """
    image = extract_single_slice_from_zarr(node, z, channel, resolution_level)
    image = convert_to_png_for_saving(image)
    iio.imwrite(f"{data_dir}/{name}_C={channel}_Z={z}.png", image)
    return image


def extract_slices_from_zarr(
    node: Node,
    start_z: int,
    num_slices: int,
    channel: int = 0,
    resolution_level: int = 0,
) -> NDArray[np.uint32]:
    """Extract a stack of 2D slices from a 3D OME-Zarr volume.

    Slice z-indices are sampled symmetrically around ``start_z`` via
    ``np.linspace(start_z - num_slices/2, start_z + num_slices/2,
    num_slices + 1, dtype=int)``. Each sampled index is read from the
    volume and written to exactly one slot of the output stack
    (``full_volume[i, :, :] = image``) — never a slice-range assignment
    that would clobber subsequent slots.

    Parameters
    ----------
    node : Node
        The OME-Zarr node to extract the slices from.
    start_z : int
        Centre z index of the slice range.
    num_slices : int
        Number of slices to extract (the output stack has
        ``num_slices + 1`` slots due to the linspace endpoint inclusion).
    channel : int
        Channel to extract (used when the volume is 4D).
    resolution_level : int
        Resolution level to extract from the multiscale pyramid.

    Returns
    -------
    NDArray[np.uint32]
        3D volume with the extracted slices (shape
        ``(num_slices + 1, H, W)``).
    """
    volume = node.data[resolution_level]
    if volume.ndim == 4:
        volume = volume[channel]

    image_zs = np.linspace(
        start_z - num_slices / 2, start_z + num_slices / 2, num_slices + 1, dtype=int
    )
    full_volume = np.zeros((len(image_zs), volume.shape[1], volume.shape[2]), dtype=np.uint32)

    for i, z in enumerate(image_zs):
        idx = int(z)
        image = volume[idx, :, :]
        image = image.compute()
        full_volume[i, :, :] = image

    return full_volume


def extract_and_save_slices_from_zarr(
    node: Node,
    start_z: int,
    num_slices: int,
    data_dir: str,
    channel: int = 0,
    resolution_level: int = 0,
    name: str = "S1",
    save_mip: bool = False,
) -> NDArray[np.uint32]:
    """Extract a stack of 2D slices from a 3D OME-Zarr volume and save them.

    The slice stack is written as a single multi-page TIFF named
    ``{name}_C={channel}_Z={lo}-{hi}.tif`` under ``data_dir``. When
    ``save_mip`` is set, a maximum-intensity projection is also written
    as ``{name}_C={channel}_Z={lo}-{hi}_mip.png``.

    Parameters
    ----------
    node : Node
        The OME-Zarr node to extract the slices from.
    start_z : int
        Centre z index of the slice range.
    num_slices : int
        Number of slices to extract (the output stack has
        ``num_slices + 1`` slots due to the linspace endpoint inclusion).
    data_dir : str
        Directory to save the slice TIFF (and optional MIP PNG) into.
    channel : int
        Channel to extract (used when the volume is 4D).
    resolution_level : int
        Resolution level to extract from the multiscale pyramid.
    name : str
        Name prefix of the volume, used in the output filename.
    save_mip : bool
        When True, also write a maximum-intensity projection as a PNG.

    Returns
    -------
    NDArray[np.uint32]
        3D volume with the extracted slices (shape
        ``(num_slices + 1, H, W)``).
    """
    volume = extract_slices_from_zarr(
        node, start_z, num_slices, channel=channel, resolution_level=resolution_level
    )

    iio.imwrite(
        f"{data_dir}/{name}_C={channel}_Z="
        f"{start_z - num_slices // 2}-{start_z + num_slices // 2}.tif",
        volume,
    )
    if save_mip:
        mip = np.max(volume, axis=0)
        mip = convert_to_png_for_saving(mip)
        iio.imwrite(
            f"{data_dir}/{name}_C={channel}_Z="
            f"{start_z - num_slices // 2}-{start_z + num_slices // 2}_mip.png",
            mip,
        )

    return volume


def colour_image(
    slice_image: ArrayLike, colour_dict: list[dict[str, int | Sequence[int]]]
) -> NDArray[np.uint8]:
    """Colour a labelled slice image based on a colour dictionary.

    Each entry in ``colour_dict`` maps a label value to an RGBA colour.
    Pixels in ``slice_image`` equal to a label's ``label-value`` are
    painted with the first three (RGB) components of that label's
    ``rgba`` entry.

    Parameters
    ----------
    slice_image : ArrayLike
        2D labelled slice where each pixel value is a label index.
    colour_dict : list[dict[str, int | Sequence[int]]]
        List of label descriptors. Each dict carries a ``"label-value"``
        int and an ``"rgba"`` sequence of int components; only the first
        three (RGB) components are used.

    Returns
    -------
    NDArray[np.uint8]
        RGB-coloured image of shape ``(H, W, 3)`` with uint8 dtype.
    """
    slice_png = np.zeros_like(slice_image, dtype="uint8")

    # Add 3rd dimension of size 3 to png
    slice_png = np.repeat(slice_png[:, :, np.newaxis], 3, axis=2)

    # Apply colour dict to image
    for i in range(len(colour_dict)):
        x, y = np.where(slice_image == colour_dict[i]["label-value"])
        slice_png[x, y, :] = colour_dict[i]["rgba"][0:3]

    return slice_png
