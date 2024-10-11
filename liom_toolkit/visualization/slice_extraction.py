import numpy as np
from skimage.io import imsave

from liom_toolkit.registration import generate_label_color_dict_allen
from liom_toolkit.utils import convert_to_png_for_saving, load_zarr, load_node_by_name


def extract_single_slice_from_zarr(zarr_file: str, z: int, channel: int = 0, resolution_level: int = 0) -> np.ndarray:
    """
    Extracts a single slice from a 3D volume

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
    :param z: Z index of the slice
    :type z: int
    :param channel: Channel to extract
    :type channel: int
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :return: The extracted slice
    :rtype: np.ndarray
    """
    nodes = load_zarr(zarr_file)
    image_node = nodes[0]
    volume = image_node.data[resolution_level]
    if volume.ndim == 4:
        volume = volume[channel]

    image = volume[z, :, :]
    image = image.compute()
    return image


def extract_and_save_slice_form_zarr(zarr_file: str, z: int, data_dir: str, channel: int = 0,
                                     resolution_level: int = 0, name: str = "S1"):
    """
    Extracts a single slice from a 3D volume and saves it to disk

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
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
    image = extract_single_slice_from_zarr(zarr_file, z, channel, resolution_level)
    image = convert_to_png_for_saving(image)
    imsave(f"{data_dir}/{name}_C={channel}_Z={z}.png", image)
    return image


def extract_slices_form_zarr(zarr_file: str, start_z: int, num_slices: int, channel=0,
                             resolution_level=0) -> np.ndarray:
    """
    Extracts slices from a 3D volume

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
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
    nodes = load_zarr(zarr_file)
    image_node = nodes[0]
    volume = image_node.data[resolution_level]
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


def extract_and_save_slices_form_zarr(zarr_file: str, start_z: int, num_slices: int, data_dir: str, channel: int = 0,
                                      resolution_level: int = 0, name: str = "S1",
                                      save_mip: bool = False) -> np.ndarray:
    """
    Extracts slices from a 3D volume and saves them to disk

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
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
    volume = extract_slices_form_zarr(zarr_file, start_z, num_slices, channel=channel,
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


def extract_label_slice(zarr_file: str, z: int, label: str, resolution_level: int = 0) -> np.ndarray:
    """
    Extracts a single slice from a 3D volume

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
    :param z: Z index of the slice
    :type z: int
    :param label: The type of label
    :type label: str
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :return: The extracted slice
    :rtype: np.ndarray
    """
    nodes = load_zarr(zarr_file)
    atlas_node = load_node_by_name(nodes, label)

    atlas = atlas_node.data[resolution_level]

    image = atlas[z, :, :]
    image = image.compute()
    return image


def save_atlas_slice(zarr_file: str, z: int, data_dir: str, resolution_level: int = 0, name: str = "S1") -> np.ndarray:
    """
    Extracts a single slice from a 3D volume and saves it to disk

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
    :param z: Z index of the slice
    :type z: int
    :param data_dir: Directory to save the slice
    :type data_dir: str
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :param name: Name of the volume
    :type name: str
    :return: The extracted slice
    :rtype: np.ndarray
    """
    image = extract_label_slice(zarr_file, z, "atlas", resolution_level)
    colour_dict = generate_label_color_dict_allen()
    image = colour_image(image, colour_dict)
    imsave(f"{data_dir}/{name}_Z={z}_atlas.png", image)

    return image


def save_vessel_slice(zarr_file: str, z: int, data_dir: str, colour_vessels: bool, resolution_level: int = 0,
                      name: str = "S1") -> np.ndarray:
    """
    Extracts a single vessel slice from a 3D volume and saves it to disk

    :param zarr_file: Path to the zarr file
    :type zarr_file: str
    :param z: Z index of the slice
    :type z: int
    :param data_dir: Directory to save the slice
    :type data_dir: str
    :param colour_vessels: Whether to colour the vessels according to the allen atlas
    :type colour_vessels: bool
    :param resolution_level: Resolution level to extract
    :type resolution_level: int
    :param name: Name of the volume
    :type name: str
    :return: The extracted slice
    :rtype: np.ndarray
    """
    image = extract_label_slice(zarr_file, z, "vessels", resolution_level)
    f_name = f"{data_dir}/{name}_Z={z}_vessels.png"
    if colour_vessels:
        colour_dict = generate_label_color_dict_allen()

        atlas_image = extract_label_slice(zarr_file, z, "atlas", resolution_level)
        coloured_vessels = image * atlas_image

        image = colour_image(coloured_vessels, colour_dict)
        f_name = f"{data_dir}/{name}_Z={z}_vessels_coloured.png"
    else:
        image = convert_to_png_for_saving(image)

    imsave(f_name, image)

    return image
