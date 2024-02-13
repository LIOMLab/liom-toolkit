import ants
import nrrd
import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader
from ome_zarr.writer import write_labels

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


def create_and_write_mask(zarr_file: str, scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128)) -> None:
    """
    Create a mask for a zarr file and write it to disk inside the labels group.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :param scales: The scales to use for the mask.
    :type scales: tuple.
    :param chunks: The chunks to use for the mask
    :type chunks: tuple
    """
    mask = create_mask_from_zarr(zarr_file)
    mask = mask.astype("int8")
    mask_transposed = np.transpose(mask, (2, 1, 0))
    save_mask_to_zarr(mask_transposed, zarr_file, scales=scales, chunks=chunks)


def create_mask_from_zarr(zarr_file: str) -> np.ndarray:
    """
    Create a brain mask from a zarr file.

    :param zarr_file: The zarr file to create a mask for.
    :type zarr_file: str
    :return: The mask
    :rtype: np.ndarray
    """
    node = load_zarr(zarr_file)[0]
    image = load_zarr_image_from_node(node, resolution_level=0)
    mask = segment_3d_brain(image)
    return mask


def save_mask_to_zarr(mask: np.ndarray, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
                      chunks: tuple = (128, 128, 128)) -> None:
    """
    Save a mask to a zarr file inside the labels group.

    :param mask: The mask to save.
    :type mask: np.ndarray
    :param zarr_file: The zarr file to save the mask to.
    :type zarr_file: str
    :param scales: The scales to use for the mask.
    :type scales: tuple
    :param chunks: The chunks to use for the mask.
    :type chunks: tuple
    """
    file = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=file)

    label_metadata = {"colors": generate_label_color_dict(),
                      "source": {
                          "image": "../../"
                      }
                      }

    write_labels(labels=mask, group=root, axes=generate_axes_dict(),
                 coordinate_transformations=create_transformation_dict(scales, 5),
                 chunks=chunks, scaler=CustomScaler(order=0, anti_aliasing=False, downscale=2, method="nearest"),
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
