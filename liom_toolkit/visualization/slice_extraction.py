import numpy as np
from ome_zarr.reader import Node
from skimage.io import imsave

from liom_toolkit.utils import convert_to_png_for_saving


def extract_single_slice_from_zarr(node: Node, z: int, channel: int = 0, resolution_level: int = 0) -> np.ndarray:
    """
    Extracts a single slice from a 3D volume

    :param node: The node to extract the slice from
    :type node: Node
    :param z: Z index of the slice
    :type z: int
    :param channel: Channel to extract
    :type channel: int
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :return: The extracted slice
    :rtype: np.ndarray
    """
    volume = node.data[resolution_level]
    if volume.ndim == 4:
        volume = volume[channel]

    image = volume[z, :, :]
    image = image.compute()
    return image


def extract_and_save_slice_form_zarr(node: Node, z: int, data_dir: str, channel: int = 0,
                                     resolution_level: int = 0, name: str = "S1"):
    """
    Extracts a single slice from a 3D volume and saves it to disk

    :param node: The node to extract the slice from
    :type node: Node
    :param z: Z index of the slice
    :type z: int
    :param data_dir: Directory to save the slice
    :type data_dir: str
    :param channel: Channel to extract
    :type channel: int
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :param name: Name of the volume
    :type name: str
    :return: The extracted slice
    :rtype: np.ndarray
    """
    image = extract_single_slice_from_zarr(node, z, channel, resolution_level)
    image = convert_to_png_for_saving(image)
    imsave(f"{data_dir}/{name}_C={channel}_Z={z}.png", image)
    return image


def extract_slices_form_zarr(node: Node, start_z: int, num_slices: int, channel=0,
                             resolution_level=0) -> np.ndarray:
    """
    Extracts slices from a 3D volume

    :param node: The node to extract the slices from
    :type node: Node
    :param start_z: Z index of the slice
    :type start_z: int
    :param num_slices: Number of slices to extract
    :type num_slices: int
    :param channel: Channel to extract
    :type channel: int
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :return: 3D volume with the extracted slices
    :rtype: np.ndarray
    """
    volume = node.data[resolution_level]
    if volume.ndim == 4:
        volume = volume[channel]

    image_zs = np.linspace(start_z - num_slices / 2, start_z + num_slices / 2, num_slices + 1, dtype=int)
    full_volume = np.zeros((len(image_zs), volume.shape[1], volume.shape[2]), dtype=np.uint32)

    for i, z in enumerate(image_zs):
        idx = int(z)
        image = volume[idx, :, :]
        image = image.compute()
        full_volume[i:, :, :] = image

    return full_volume


def extract_and_save_slices_form_zarr(node: Node, start_z: int, num_slices: int, data_dir: str, channel: int = 0,
                                      resolution_level: int = 0, name: str = "S1",
                                      save_mip: bool = False) -> np.ndarray:
    """
    Extracts slices from a 3D volume and saves them to disk

    :param node: The node to extract the slices from
    :type node: Node
    :param start_z: Z index of the slice
    :type start_z: int
    :param num_slices: Number of slices to extract
    :type num_slices: int
    :param data_dir: Directory to save the slices
    :type data_dir: str
    :param channel: Channel to extract
    :type channel: int
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :param name: Name of the volume
    :type name: str
    :param save_mip: Whether to save the maximum intensity projection
    :type save_mip: bool
    :return: 3D volume with the extracted slices
    :rtype: np.ndarray
    """
    volume = extract_slices_form_zarr(node, start_z, num_slices, channel=channel,
                                      resolution_level=resolution_level)

    imsave(f"{data_dir}/{name}_C={channel}_Z={start_z - num_slices // 2}-{start_z + num_slices // 2}.tif", volume)
    if save_mip:
        mip = np.max(volume, axis=0)
        mip = convert_to_png_for_saving(mip)
        imsave(f"{data_dir}/{name}_C={channel}_Z={start_z - num_slices // 2}-{start_z + num_slices // 2}_mip.png", mip)

    return volume


def colour_image(slice_image: np.ndarray, colour_dict: list):
    """
    Colour an image based on a colour dictionary

    :param slice_image: The slice to colour
    :type slice_image: np.ndarray
    :param colour_dict: The colour dictionary
    :type colour_dict: list
    :return: The coloured image
    :rtype: np.ndarray
    """
    slice_png = np.zeros_like(slice_image, dtype='uint8')

    # Add 3rd dimension of size 3 to png
    slice_png = np.repeat(slice_png[:, :, np.newaxis], 3, axis=2)

    # Apply colour dict to image
    for i in range(len(colour_dict)):
        x, y = np.where(slice_image == colour_dict[i]['label-value'])
        slice_png[x, y, :] = colour_dict[i]['rgba'][0:3]

    return slice_png
