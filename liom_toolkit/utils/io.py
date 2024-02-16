import tempfile

import ants
import nrrd
import numpy as np
import zarr
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader
from ome_zarr.writer import write_labels

from liom_toolkit.registration import download_allen_atlas
from liom_toolkit.segmentation import segment_3d_brain
from liom_toolkit.utils import generate_axes_dict, create_transformation_dict, CustomScaler


def load_zarr(zarr_file: str) -> list[Node]:
    """
    Load a zarr file to an ANTs image.

    :param zarr_file: The zarr file to load.
    :type zarr_file: str
    :return: The ANTs image.
    :rtype: ants.ANTsImage
    """
    reader = Reader(parse_url(zarr_file))
    image_node = list(reader())
    return image_node


def load_zarr_image_from_node(node: Node, resolution_level: int = 1,
                              volume_direction: tuple = ([1., 0., 0.], [0., 0., -1.], [0., -1., 0.])) -> ants.ANTsImage:
    """
    Load a zarr file to an ANTs image.

    :param node: The zarr node to load.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :param volume_direction: The direction of the volume.
    :type volume_direction: tuple
    :return: The ANTs image.
    :rtype: ants.ANTsImage
    """
    volume = np.array(node.data[resolution_level])
    volume = np.transpose(volume, (2, 1, 0)).astype("uint32")
    volume = ants.from_numpy(volume)
    transform = load_zarr_transform_from_node(node, resolution_level=resolution_level)

    # Convert to mm
    transform = [element / 1000 for element in transform]
    volume.set_spacing(transform)
    volume.set_direction(volume_direction)
    volume.physical_shape = tuple(np.array(volume.shape) * np.array(volume.spacing))
    return volume


def load_allen_template(atlas_file: str, resolution: int, padding: bool) -> ants.ANTsImage:
    """
    Load the allen template and set the resolution and direction (PIR).

    :param atlas_file: The file to load.
    :type atlas_file: str
    :param resolution: The resolution to set.
    :type resolution: int
    :param padding: Whether to pad the atlas or not.
    :type padding: bool
    :return: The loaded template.
    :rtype: ants.ANTsImage
    """
    resolution = resolution / 1000
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
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :return: The coordinate transform matching the resolution level.
    :rtype: ants.ANTsImage
    """
    transform = node.metadata["coordinateTransformations"][resolution_level][0]['scale']
    return transform


def save_atlas_to_zarr(zarr_file: str, atlas: ants.ANTsImage, scales: tuple = (6.5, 6.5, 6.5),
                       chunks: tuple = (128, 128, 128), resolution_level: int = 0) -> None:
    """
    Save an atlas to a zarr file inside the labels group.

    :param zarr_file: The zarr file to save the atlas to.
    :type zarr_file: str
    :param atlas: The atlas to save.
    :type atlas: ants.ANTsImage
    :param scales: The scales to use for the atlas.
    :type scales: tuple
    :param chunks: The chunks to use for the atlas.
    :type chunks: tuple
    :param resolution_level: The resolution level of the atlas.
    :type resolution_level: int
    """

    color_dict = generate_label_color_dict_allen()
    save_annotation_to_zarr(atlas.numpy(), zarr_file, color_dict, scales, chunks, resolution_level)


def create_and_write_mask(zarr_file: str, scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128),
                          resolution_level: int = 0) -> None:
    """
    Create a mask for a zarr file and write it to disk inside the labels group.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :param scales: The scales to use for the mask.
    :type scales: tuple.
    :param chunks: The chunks to use for the mask
    :type chunks: tuple
    :param resolution_level: The resolution level of the mask.
    :type resolution_level: int
    """
    mask = create_mask_from_zarr(zarr_file, resolution_level)
    mask = mask.astype("int8")
    mask_transposed = np.transpose(mask, (2, 1, 0))
    color_dict = generate_label_color_dict()
    save_annotation_to_zarr(mask_transposed, zarr_file, scales=scales, chunks=chunks, color_dict=color_dict)


def create_mask_from_zarr(zarr_file: str, resolution_level: int = 0) -> np.ndarray:
    """
    Create a brain mask from a zarr file.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :param resolution_level: The resolution level of the mask.
    :type resolution_level: int
    :return: The mask
    :rtype: np.ndarray
    """
    node = load_zarr(zarr_file)[0]
    image = load_zarr_image_from_node(node, resolution_level=resolution_level)
    mask = segment_3d_brain(image)
    return mask


def save_annotation_to_zarr(mask: np.ndarray, zarr_file: str, color_dict: list[dict], scales: tuple = (6.5, 6.5, 6.5),
                            chunks: tuple = (128, 128, 128), resolution_level: int = 0) -> None:
    """
    Save a mask to a zarr file inside the labels group.

    :param mask: The mask to save.
    :type mask: np.ndarray
    :param zarr_file: The zarr file to save the mask to.
    :type zarr_file: str
    :param color_dict: The color dictionary to use for the mask.
    :type color_dict: list[dict]
    :param scales: The scales to use for the mask.
    :type scales: tuple
    :param chunks: The chunks to use for the mask.
    :type chunks: tuple
    :param resolution_level: The resolution level of the mask.
    :type resolution_level: int
    """
    file = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=file)

    label_metadata = {"colors": color_dict,
                      "source": {
                          "image": "../../"
                      }
                      }

    write_labels(labels=mask, group=root, axes=generate_axes_dict(),
                 coordinate_transformations=create_transformation_dict(scales, 5),
                 chunks=chunks, scaler=CustomScaler(order=0, anti_aliasing=False, downscale=2, method="nearest",
                                                    input_layer=resolution_level),
                 name="mask", label_metadata=label_metadata)


def set_physical_shape(image: ants.ANTsImage) -> None:
    """
    Set the physical shape of an ANTs image by multiplying the shape with the spacing.

    :param image: The image to set the physical shape for.
    :type image: ants.ANTsImage
    """
    dims = np.array(image.shape)
    spacing = np.array(image.spacing)
    physical_shape = tuple(dims * spacing)
    image.physical_shape = physical_shape


def generate_label_color_dict() -> list[dict]:
    """
    Generate a label color dictionary for the mask. Black is background, green is foreground.

    :return: The label color dictionary.
    :rtype: list[dict]
    """
    label_colors = [
        {
            "label-value": 0,
            "rgba": [0, 0, 0, 0]
        },
        {
            "label-value": 1,
            "rgba": [0, 255, 0, 64]
        }
    ]
    return label_colors


def generate_label_color_dict_allen() -> list[dict]:
    """
    Generate a label color dictionary for the allen atlas.

    :return: The label color dictionary.
    :rtype: list[dict]
    """
    temp_dir = tempfile.TemporaryDirectory()
    atlas = download_allen_atlas(temp_dir.name, 25)
    atlas = atlas.numpy()

    # Grab the structure tree instance
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()

    # Get a list of structures inside the slice
    structure_id_list = np.unique(atlas.ravel()).tolist()
    structure_id_list.remove(0)  # Remove the background
    structures = structure_tree.get_structures_by_id(structure_id_list)

    # Generate a color dictionary for the input atlas image
    color_dict = []
    for structure_id, structure in zip(structure_id_list, structures):
        if structure is None:
            continue
        color = structure['rgb_triplet']
        color.append(255)
        color_dict.append({"label-value": structure_id, "rgba": color})

    temp_dir.cleanup()
    return color_dict
