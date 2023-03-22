import ants
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node

"""
    Utility functions aiding in the registration process.
"""


def load_zarr(zarr_file: str) -> Node:
    """
    Load a zarr file to an ANTs image.

    :param zarr_file: The zarr file to load.
    :return: The ANTs image.
    """
    reader = Reader(parse_url(zarr_file))
    image_node = list(reader())[0]
    return image_node


def load_zarr_image_from_node(node: Node, scale: (list or tuple), resolution_level: int = 1) -> ants.ANTsImage:
    """
    Load a zarr file to an ANTs image.

    :param node: The zarr node to load.
    :param scale: The scale of the image.
    :param resolution_level: The resolution level to load.
    :return: The ANTs image.
    """
    volume = np.array(node.data[resolution_level])
    volume = np.transpose(volume, (2, 1, 0)).astype("uint32")
    volume = ants.from_numpy(volume)
    volume.set_spacing(scale)
    volume.set_direction([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    return volume


def load_zarr_transform_from_node(node: Node, resolution_level: int = 1) -> dict:
    """
    Load a zarr file to an ANTs image.

    :param node: The zarr node to load.
    :param resolution_level: The resolution level to load.
    :return: The ANTs image.
    """
    transform = node.metadata["coordinateTransformations"][resolution_level][0]['scale']
    return transform


def load_ants_image_from_zarr(node: Node, resolution_level: int = 1) -> ants.ANTsImage:
    """
    Load a zarr file to an ANTs image.

    :param node: The zarr node to load.
    :param resolution_level: The resolution level to load.
    :return: The ANTs image.
    """
    transform = load_zarr_transform_from_node(node, resolution_level=resolution_level)
    volume = load_zarr_image_from_node(node, scale=transform, resolution_level=resolution_level)
    return volume
