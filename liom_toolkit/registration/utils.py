from typing import List, Tuple

import SimpleITK as sitk
import ants
import nrrd
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader, Node
from ome_zarr.writer import write_image
from scipy.ndimage import binary_fill_holes

from liom_toolkit.segmentation import remove_small_structures
from liom_toolkit.utils import generate_axes_dict, create_transformation_dict, CustomScaler

"""
    Utility functions aiding in the registration process.
"""


def load_zarr(zarr_file: str) -> List[Node]:
    """
    Load a zarr file to an ANTs image.

    :param zarr_file: The zarr file to load.
    :return: The ANTs image.
    """
    reader = Reader(parse_url(zarr_file))
    image_node = list(reader())
    return image_node


def load_zarr_image_from_node(node: Node, scale: (List or Tuple), resolution_level: int = 1) -> ants.ANTsImage:
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
    volume.set_direction([[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    return volume


def load_mask(node: Node, resolution_level: int = 0) -> ants.ANTsImage:
    """
    Load a mask from a zarr file.

    :param node: The zarr node to load the mask from
    :param resolution_level: The resolution level to load the mask from
    :return: The loaded mask
    """
    volume = np.array(node.data[resolution_level])
    volume = np.transpose(volume, (2, 1, 0)).astype("ubyte")
    mask = ants.from_numpy(volume)

    transform = load_zarr_transform_from_node(node, resolution_level=resolution_level)
    mask.set_spacing(transform)
    mask.set_direction([[1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    return mask


def load_allen_template(atlas_file: str, resolution: int, padding: bool) -> ants.ANTsImage:
    """
    Load the allen template and set the resolution and direction (PIR).

    :param atlas_file: The file to load
    :param resolution: The resolution to set
    :param padding: Whether to pad the atlas or not
    :return: The loaded template
    """
    atlas_data, atlas_header = nrrd.read(atlas_file)
    atlas_data = atlas_data.astype("uint32")
    if padding:
        # Pad the atlas to avoid edge effects, the padding is 15% of the atlas size
        pad_size = int(atlas_data.shape[0] * 0.15)
        npad = ((pad_size, pad_size), (0, 0), (0, 0))
        atlas_data = np.pad(atlas_data, pad_width=npad, mode="constant", constant_values=0)
    atlas_volume = ants.from_numpy(atlas_data)
    atlas_volume.set_spacing([resolution, resolution, resolution])
    atlas_volume.set_direction([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
    return atlas_volume


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


def fill_holes_2d_3d(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a 2D and 3D mask.

    Source: https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    :param mask: The mask to fill holes in
    :return: The mask with holes filled
    """
    # Filling holes and returning the mask
    mask = binary_fill_holes(mask)

    # Fill holes (in 2D)
    nx, ny, nz = mask.shape
    for x in range(nx):
        mask[x, :, :] = binary_fill_holes(mask[x, :, :])
    for y in range(ny):
        mask[:, y, :] = binary_fill_holes(mask[:, y, :])
    for z in range(nz):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])

    # Refill holes in 3D (in case some were missed)
    mask = binary_fill_holes(mask)
    return mask


def segment_3d_brain(volume: ants.ANTsImage, k=5, useLog=True, thresholdMethod="otsu") -> np.ndarray:
    """
    Segment a 3D brain volume using a watershed algorithm.

    Source: https://github.com/linum-uqam/sbh-reconstruction/blob/51271c84347afccb21483cfd3fcbde77d537929c/slicercode/segmentation/brainMask.py

    :param volume: The volume to segment
    :param k: The size of the median filter
    :param useLog: Whether to use the log of the volume
    :param thresholdMethod: The threshold method to use
    :return: The segmented mask
    """
    np_volume = volume.numpy()
    vol_p = np.copy(np_volume)
    if useLog:
        vol_p[np_volume > 0] = np.log(vol_p[np_volume > 0])

    # Creating a sitk image + smoothing
    img = sitk.GetImageFromArray(vol_p)
    img = sitk.Median(img, [k, k, k])

    # Segmenting using an Otsu threshold
    if thresholdMethod == "otsu":
        marker_img = ~sitk.OtsuThreshold(img)
    elif thresholdMethod == "triangle":
        marker_img = ~sitk.TriangleThreshold(img)
    else:
        marker_img = ~sitk.OtsuThreshold(img)

    # Using a watershed algorithm to optimize the mask
    ws = sitk.MorphologicalWatershedFromMarkers(img, marker_img)

    # Separating into foreground / background
    seg = sitk.ConnectedComponent(ws != ws[0, 0, 0])

    # Filling holes and returning the mask
    mask = fill_holes_2d_3d(sitk.GetArrayFromImage(seg))

    # Remove small objects
    mask = remove_small_structures(vol_p, mask)

    return mask


def create_and_write_mask(zarr_file: str, overwrite: bool = False):
    """
    Create a mask for a zarr file and write it to disk.

    :param zarr_file: The zarr file to create a mask for
    :param overwrite: Whether to overwrite the mask if it already exists
    """
    node = load_zarr(zarr_file)[0]
    image = load_ants_image_from_zarr(node, resolution_level=0)
    mask = segment_3d_brain(image)
    mask_transposed = np.transpose(mask, (2, 1, 0))

    file = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=file)
    labels_grp = root.create_group("labels", overwrite=overwrite)
    label_name = "mask"
    labels_grp.attrs["labels"] = [label_name]
    label_grp = labels_grp.create_group(label_name, overwrite=overwrite)

    write_image(image=mask_transposed, group=label_grp, axes=generate_axes_dict(),
                coordinate_transformations=create_transformation_dict((6.5, 6.5, 6.5), 5),
                storage_options=dict(chunks=(512, 512, 512)), scaler=CustomScaler(downscale=2, method="nearest"))
