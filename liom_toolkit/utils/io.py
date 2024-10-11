import os
import tempfile
from typing import Callable, Union

import ants
import dask.array as da
import nrrd
import numpy as np
import zarr
from ants.core.ants_image import ANTsImage
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader
from ome_zarr.scale import Scaler, ArrayLike
from ome_zarr.writer import write_labels
from skimage.io import imsave
from skimage.transform import resize
from tqdm.auto import tqdm

from liom_toolkit.registration import download_allen_atlas
from liom_toolkit.segmentation import segment_3d_brain
from .utils import convert_to_png_for_saving


def load_zarr(zarr_file: str) -> list[Node]:
    """
    Load a zarr file to an ANTs image.

    :param zarr_file: The zarr file to load.
    :type zarr_file: str
    :return: The loaded zarr file.
    :rtype: list[Node]
    """
    reader = Reader(parse_url(zarr_file))
    nodes = list(reader())
    return nodes


def load_zarr_image_from_node(node: Node, resolution_level: int = 1) -> da.array:
    """
    Load a zarr file to an ANTs image. Loads one channel at a time.

    :param node: The zarr node to load.
    :type node: Node
    :param resolution_level: The resolution level to load.
    :type resolution_level: int
    :return: The image.
    :rtype: da.array
    """
    volume = node.data[resolution_level]
    return volume


def convert_dask_to_ants(dask_array: da.Array, node: Node, resolution_level: int = 2,
                         volume_direction: tuple = ([1., 0., 0.], [0., 0., -1.], [0., -1., 0.])) -> ANTsImage:
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
    image = load_zarr_image_from_node(node, resolution_level)
    if len(image.shape) == 4:
        image = image[channel, :, :, :]
    ants_image = convert_dask_to_ants(image, node, resolution_level)
    return ants_image


def load_allen_template(atlas_file: str, resolution: int, padding: bool) -> ANTsImage:
    """
    Load the allen template and set the resolution and direction (PIR).

    :param atlas_file: The file to load.
    :type atlas_file: str
    :param resolution: The resolution to set.
    :type resolution: int
    :param padding: Whether to pad the atlas or not.
    :type padding: bool
    :return: The loaded template.
    :rtype: ANTsImage
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
    :rtype: ANTsImage
    """
    transform = node.metadata["coordinateTransformations"][resolution_level][0]['scale']
    return transform


def save_atlas_to_zarr(zarr_file: str, atlas: ArrayLike, scales: tuple = (6.5, 6.5, 6.5),
                       chunks: tuple = (128, 128, 128), resolution_level: int = 0) -> None:
    """
    Save an atlas to a zarr file inside the labels group.

    :param zarr_file: The zarr file to save the atlas to.
    :type zarr_file: str
    :param atlas: The atlas to save.
    :type atlas: ArrayLike
    :param scales: The scales to use for the atlas.
    :type scales: tuple
    :param chunks: The chunks to use for the atlas.
    :type chunks: tuple
    :param resolution_level: The resolution level of the atlas.
    :type resolution_level: int
    """
    color_dict = generate_label_color_dict_allen()
    save_label_to_zarr(label=atlas, zarr_file=zarr_file, color_dict=color_dict, scales=scales, chunks=chunks,
                       resolution_level=resolution_level, name="atlas")


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
    color_dict = generate_label_color_dict_mask()
    save_label_to_zarr(mask_transposed, zarr_file, scales=scales, chunks=chunks, color_dict=color_dict,
                       name="mask", resolution_level=resolution_level)


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
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    image = image.compute()
    mask = segment_3d_brain(image)
    return mask


def save_label_to_zarr(label: ArrayLike, zarr_file: str, color_dict: list[dict], name: str,
                       scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128),
                       resolution_level: int = 0, ) -> None:
    """
    Save a mask to a zarr file inside the labels group.

    :param label: The mask to save.
    :type label: np.ndarray
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
    :param name: The name of the mask.
    :type name: str
    """
    n_dims = len(label.shape)
    file = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=file)

    label_metadata = {"colors": color_dict,
                      "source": {
                          "image": "../../"
                      }
                      }

    if isinstance(label, da.Array):
        scaler = Scaler()
    else:
        scaler = CustomScaler(order=0, anti_aliasing=False, downscale=2, method="nearest",
                              input_layer=resolution_level, original_image=zarr_file)

    write_labels(labels=label, group=root, axes=generate_axes_dict(n_dims),
                 coordinate_transformations=create_transformation_dict(scales, 5, n_dims),
                 chunks=chunks, scaler=scaler, name=name, label_metadata=label_metadata)


def set_physical_shape(image: ANTsImage) -> None:
    """
    Set the physical shape of an ANTs image by multiplying the shape with the spacing.

    :param image: The image to set the physical shape for.
    :type image: ANTsImage
    """
    dims = np.array(image.shape)
    spacing = np.array(image.spacing)
    physical_shape = tuple(dims * spacing)
    image.physical_shape = physical_shape


def generate_label_color_dict_mask() -> list[dict]:
    """
    Generate a label color dictionary for the mask. Black is background, white is foreground.

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
            "rgba": [255, 255, 255, 255]
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

    annotation, meta = download_allen_atlas(temp_dir.name, resolution=25, keep_nrrd=False)

    # Generate a color dictionary according to the OME-NGFF specification
    color_dict = []
    for row in meta.iterrows():
        color_dict.append({"label-value": row[1]['IDX'],
                           "rgba": [row[1]['-R-'], row[1]['-G-'], row[1]['-B-'], (int(row[1]['-A-'] * 255))]})

    temp_dir.cleanup()
    return color_dict


class CustomScaler(Scaler):
    """
    A custom scaler that can down-sample 3D images for OME-Zarr Conversion.

    :param order: The order of the transformation.
    :type order: int
    :param anti_aliasing: Whether to use anti-aliasing
    :type anti_aliasing: bool
    :param downscale: The amount to downscale by
    :type downscale: int
    :param method: The method to use for downscaling. **Disclaimer:** Only "nearest" is supported.
    :type method: str
    :param input_layer: The input layer to use for the transformation.
    :type input_layer: int
    :param max_layer: The maximum layer to use for the transformation.
    :type max_layer: int
    :param original_image: The original image to use for the transformation.
    :type original_image: str | None
    """
    order: int
    anti_aliasing: bool
    input_layer: int
    to_down_scale: np.ndarray
    to_up_scale: np.ndarray
    do_upscale: bool = True
    original_image: str | None
    current_scale: int = None

    def __init__(self, order: int = 1, anti_aliasing: bool = True, downscale: int = 2, method: str = "nearest",
                 input_layer: int = 0, max_layer: int = 4, original_image: str | None = None):
        super().__init__(downscale=downscale, method=method, max_layer=max_layer)
        self.order = order
        self.anti_aliasing = anti_aliasing
        self.input_layer = input_layer
        self.original_image = original_image

    def nearest(self, base: np.ndarray) -> list[np.ndarray]:
        """
        Down-sample using :func:`skimage.transform.resize`.

        :param base: The base image to down-sample.
        :type base: np.ndarray
        :return: The down-sampled image.
        :rtype: list[np.ndarray]
        """
        # Determine the levels to scale and in which direction
        scales = np.linspace(0, self.max_layer, self.max_layer + 1, dtype=int)
        base_layer = self.input_layer

        self.to_up_scale = scales[:base_layer]
        self.to_down_scale = scales[base_layer + 1:]

        if len(self.to_up_scale) == 0:
            self.do_upscale = False

        return self._by_plane(base, self.__nearest)

    def __nearest(self, plane: ArrayLike, size_y: int, size_x: int) -> np.ndarray:
        """Apply the 2-dimensional transformation.

        :param plane: The plane to transform.
        :type plane: ArrayLike
        :param size_y: The size of the y dimension.
        :type size_y: int
        :param size_x: The size of the x dimension.
        :type size_x: int
        :return: The transformed plane.
        :rtype: np.ndarray
        """
        if isinstance(plane, da.Array):

            def _resize(
                    image: ArrayLike, output_shape: tuple, **kwargs
            ) -> ArrayLike:
                return dask_resize(image, output_shape, **kwargs)

        else:
            _resize = resize

        if self.do_upscale:
            shape = plane.shape[0] * self.downscale, plane.shape[1] * self.downscale, plane.shape[
                2] * self.downscale
            if self.original_image is not None:
                nodes = load_zarr(self.original_image)
                image_node = nodes[0]
                image = image_node.data[self.current_scale]
                shape = image.shape
                del image
                del image_node
                del nodes
                if len(shape) == 4:
                    shape = shape[1:]
            output_shape = shape
        else:
            output_shape = plane.shape[0] // self.downscale, plane.shape[1] // self.downscale, plane.shape[
                2] // self.downscale

        return _resize(
            plane,
            output_shape=output_shape,
            order=self.order,
            preserve_range=True,
            anti_aliasing=self.anti_aliasing,
        ).astype(plane.dtype)

    def _by_plane(
            self,
            base: np.ndarray,
            func: Callable[[np.ndarray, int, int], np.ndarray],
    ) -> list[Union[np.ndarray, np.ndarray, None]]:
        """Loop over 3 of the 5 dimensions and apply the func transform.

        :param base: The base image to transform.
        :type base: np.ndarray
        :param func: The function to apply to the image.
        :type func: Callable[[np.ndarray, int, int], np.ndarray]
        :return: The transformed image.
        :rtype: list[Union[ndarray, ndarray, None]]
        """
        aa = self.anti_aliasing
        start_scale = self.input_layer
        rv = [None] * (self.max_layer + 1)
        rv[start_scale] = base

        # Do up-scaling first
        self.anti_aliasing = False
        self.do_upscale = True
        scales = self.to_up_scale
        if scales.size > 0:
            for scale in np.flip(scales):
                self.current_scale = scale
                stack_to_scale = rv[scale + 1]
                rv[scale] = self._scale_by_plane(base, stack_to_scale, func)

        # Do down-scaling
        self.anti_aliasing = aa
        self.do_upscale = False
        scales = self.to_down_scale
        if scales.size > 0:
            for scale in scales:
                stack_to_scale = rv[scale - 1]
                rv[scale] = self._scale_by_plane(base, stack_to_scale, func)

        return rv

    def _scale_by_plane(self, base, stack_to_scale, func):
        shape_5d = (*(1,) * (5 - stack_to_scale.ndim), *stack_to_scale.shape)
        T, C, Z, Y, X = shape_5d

        # If our data is already 2D, simply resize and add to pyramid
        if stack_to_scale.ndim == 2:
            image = func(stack_to_scale, Y, X)
            return image

        # stack_dims is any dims over 3D
        new_stack = None
        for t in range(T):
            for c in range(C):
                if C > 1:
                    plane = stack_to_scale[c]
                else:
                    plane = stack_to_scale[:]
                out = func(plane, Y, X)
                # first iteration of loop creates the new nd stack
                if new_stack is None:
                    if C > 1:
                        new_stack = np.zeros(
                            (C, out.shape[0], out.shape[1], out.shape[2]),
                            dtype=base.dtype,
                        )
                    else:
                        new_stack = np.zeros(
                            (out.shape[0], out.shape[1], out.shape[2]),
                            dtype=base.dtype,
                        )
                # insert resized plane into the stack at correct indices
                if C > 1:
                    new_stack[c] = out
                else:
                    new_stack[:] = out
        image = new_stack
        return image


def create_transformation_dict(scales: tuple, levels: int, dimensions: int) -> list:
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :type scales: tuple
    :param levels: The number of levels in the pyramid.
    :type levels: int
    :param dimensions: The number of dimensions in the image.
    :type dimensions: int
    :return: The transformation dictionary.
    :rtype: list
    """
    coord_transforms = []
    for i in range(levels):
        if dimensions == 4:
            transform_dict = [{
                "type": "scale",
                "scale": [1, scales[0] * (2 ** i), scales[1] * (2 ** i), scales[2] * (2 ** i)]
            }]
        else:
            transform_dict = [{
                "type": "scale",
                "scale": [scales[0] * (2 ** i), scales[1] * (2 ** i), scales[2] * (2 ** i)]
            }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(dimensions: int) -> list:
    """
    Generate the axes dictionary for the zarr file.

    :param dimensions: The number of dimensions in the image.
    :type dimensions: int

    :return: The axes dictionary.
    :rtype: list
    """
    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
    ]
    if dimensions == 4:
        axes.insert(0, {"name": "c", "type": "channel"})
    return axes


def load_node_by_name(nodes: list[Node], name: str) -> Node | None:
    """
    Load a node by name from a zarr file. Returns None if the node is not found.

    :param nodes: The nodes to search through.
    :type nodes: list[Node]
    :param name: The name of the node to load.
    :type name: str
    :return: The loaded node.
    :rtype: Node | None
    """
    for node in nodes:
        # Check for empy dict
        if node.metadata == {}:
            continue

        if node.metadata["name"] == name:
            return node
    return None


def extract_zarr_to_png(zarr_file: str, target_dir: str, channel: int) -> None:
    """
    Extract a zarr file to a directory of PNG images.

    :param zarr_file: The zarr file to extract.
    :type zarr_file: str
    :param target_dir: The directory to save the PNG images to.
    :type target_dir: str
    :param channel: The channel to extract.
    :type channel: int
    :return: None
    """
    node = load_zarr(zarr_file)[0]
    volume = node.data[0]

    if len(volume.shape) == 4:
        volume = volume[channel]

    # Create if not exists, empty if exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        for file in os.listdir(target_dir):
            os.remove(os.path.join(target_dir, file))

    for z in tqdm(range(volume.shape[0])):
        image = volume[z, :, :]
        image = convert_to_png_for_saving(image)
        imsave(f"{target_dir}/{str(z)}.png", image, check_contrast=False)
