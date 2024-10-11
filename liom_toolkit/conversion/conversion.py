import os
import tempfile

import ants
import dask.array as da
import h5py
import nibabel as nib
import nrrd
import numpy as np
import zarr
from natsort import natsorted
from ome_zarr.dask_utils import resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, ArrayLike
from tqdm.auto import tqdm

from liom_toolkit.registration import align_annotations_to_volume
from liom_toolkit.utils.dask_client import dask_client_manager
from liom_toolkit.utils.io import load_zarr, save_atlas_to_zarr, \
    CustomScaler, create_transformation_dict, generate_axes_dict, create_mask_from_zarr, save_label_to_zarr, \
    generate_label_color_dict_mask, load_node_by_name, load_ants_image_from_node, load_zarr_image_from_node


def load_hdf5(hdf5_file: str) -> da.Array:
    """
    Load the data from a HDF5 file. If use_mem_map is True, the data will be saved to a memmap file to save memory.

    :param hdf5_file: The HDF5 file to load.
    :type hdf5_file: str
    :return: The data from the HDF5 file.
    :rtype: da.Array
    """
    client = dask_client_manager.get_client()

    f = h5py.File(hdf5_file, "r")
    keys = natsorted(list(f.keys()))
    paths = [f"/{key}" for key in keys]
    data_list = [client.submit(da.from_array, (f[path])) for path in paths]
    loaded_data = [client.gather(data) for data in data_list]
    data = da.stack(loaded_data, axis=0)
    data = da.rechunk(data, chunks=(128, 128, 128))
    data = client.persist(data)

    # Clean up
    del data_list
    del loaded_data

    return data


def convert_hdf5_to_nifti(hdf5_file: str, nifti_file: str) -> None:
    """
    Convert a HDF5 file to a NIFTI file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param nifti_file: Path to the NIFTI file.
    :type nifti_file: str
    """
    data = load_hdf5(hdf5_file)

    print("Saving...")
    data = data.compute()
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")


def save_zarr(data: ArrayLike, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
              chunks: tuple = (128, 128, 128)) -> None:
    """
    Save a numpy array to a zarr file.

    :param data: The data to save.
    :type data: np.ndarray
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    """
    n_dims = len(data.shape)

    print("Saving...")
    os.mkdir(zarr_file)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)

    if isinstance(data, da.Array):
        scaler = Scaler()
    else:
        scaler = CustomScaler(order=1, anti_aliasing=True, downscale=2, method="nearest", input_layer=0)

    write_image(image=data, group=root, axes=generate_axes_dict(n_dims),
                coordinate_transformations=create_transformation_dict(scales, 5, n_dims),
                storage_options=dict(chunks=chunks),
                scaler=scaler)
    print("Done!")


def convert_hdf5_to_zarr(hdf5_file: str, zarr_file: str, use_memmap: bool = True, remove_stripes: bool = False,
                         scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128),
                         base_key: str = "reconstructed_frame") -> None:
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.

    :param hdf5_file: Path to the HDF5 file.
    :type hdf5_file: str
    :param zarr_file: Path to the zarr file.
    :type zarr_file: str
    :param use_memmap: Whether to use a memmap or not.
    :type use_memmap: bool
    :param remove_stripes: Whether to remove stripes from the data.
    :type remove_stripes: bool
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chunks: The chunk size to use.
    :type chunks: tuple
    :param base_key: The base key of the HDF5 key list.
    :type base_key: str
    """

    map_file = "temp.dat"
    data = load_hdf5(hdf5_file)

    save_zarr(data, zarr_file, scales=scales, chunks=chunks)
    if use_memmap:
        os.remove(map_file)


def convert_nifti_to_zarr(nifti_file: str, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
                          chucks: tuple = (128, 128, 128), transpose: bool = False) -> None:
    """
    Convert a NIFTI file to a zarr file.

    :param nifti_file: The NIFTI file to convert.
    :type nifti_file: str
    :param zarr_file: The zarr file to save to.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chucks: The chunk size to use in the zarr file.
    :type chucks: tuple
    :param transpose: Whether to transpose the data or not.
    :type transpose: bool
    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = da.from_array(ni_img.get_fdata())
    if transpose:
        data = da.transpose(data, (2, 1, 0))
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)


def convert_nrrd_to_zarr(nrrd_file: str, zarr_file: str, scales: tuple = (6.5, 6.5, 6.5),
                         chucks: tuple = (128, 128, 128)) -> None:
    """
    Convert a NRRD file to a zarr file.

    :param nrrd_file: The NRRD file to convert.
    :type nrrd_file: str
    :param zarr_file: The zarr file to save.
    :type zarr_file: str
    :param scales: The resolution of the image, in z y x order.
    :type scales: tuple
    :param chucks: The chunk size to use in the zarr file.
    :type chucks: tuple
    """
    print("Loading...")
    data, header = nrrd.read(nrrd_file)
    save_zarr(data, zarr_file, scales=scales, chunks=chucks)


def create_multichannel_zarr(auto_fluo_file: str, vascular_file: str, zarr_file: str,
                             scales: tuple = (6.5, 6.5, 6.5), chunks: tuple = (128, 128, 128)) -> None:
    """
    Create a multichannel zarr file from the auto-fluorescence and vascular data.

    :param auto_fluo_file: The path to the auto-fluorescence hdf5 file.
    :type auto_fluo_file: str
    :param vascular_file: The path to the vascular hdf5 file.
    :type vascular_file: str
    :param zarr_file: The path to the zarr file to save the volume to.
    :type zarr_file: str
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    :return:
    """
    client = dask_client_manager.get_client()
    # Extract data from the hdf5 files
    auto_fluo = load_hdf5(auto_fluo_file)
    vascular = load_hdf5(vascular_file)

    # Merge the data along a new fourth dimension at index 0
    volume = client.submit(da.stack, [auto_fluo, vascular], axis=0).result()
    volume = client.gather(volume)
    volume = volume.compute()

    # Save the volume to a zarr file
    save_zarr(volume, zarr_file, scales=scales, chunks=chunks)
    del auto_fluo, vascular, volume


def create_full_zarr_volume(auto_fluo_file: str, vascular_file: str, zarr_file: str, template_path: str,
                            atlas_path: str, use_custom_atlas=True, scales: tuple = (6.5, 6.5, 6.5),
                            chunks: tuple = (128, 128, 128)) -> None:
    """
    Create a full zarr volume from the auto-fluorescence and vascular data. The annotations will be aligned to the
    auto-fluorescence data and saved to the zarr file. The mask will also be created and saved to the zarr file.

    :param auto_fluo_file: The path to the auto-fluorescence hdf5 file.
    :type auto_fluo_file: str
    :param vascular_file: The path to the vascular hdf5 file.
    :type vascular_file: str
    :param zarr_file: The path to the zarr file to save the volume to.
    :type zarr_file: str
    :param template_path: The path to the template to align the annotations to.
    :type template_path: str
    :param atlas_path: The path to the atlas to use for the annotations.
    :type atlas_path: str
    :param use_custom_atlas: Whether to use a custom atlas or not.
    :type use_custom_atlas: bool
    :param scales: The physical resolution of the volume per axis.
    :type scales: tuple
    :param chunks: The chunk size to use for the volume.
    :type chunks: tuple
    """
    temp_dir = tempfile.TemporaryDirectory()
    resolution_level = 2
    atlas_resolution = 25

    pbar = tqdm(total=5, desc="Creating zarr volume")
    pbar.set_postfix({"step": "Creating multichannel zarr"})
    create_multichannel_zarr(auto_fluo_file, vascular_file, zarr_file, scales=scales, chunks=chunks)
    pbar.update(1)

    pbar.set_postfix({"step": "Creating temporary mask"})
    # Load image for image information
    nodes = load_zarr(zarr_file)
    target_image = load_ants_image_from_node(nodes[0], resolution_level, channel=0)
    # Create the temporary mask
    mask = create_mask_from_zarr(zarr_file, resolution_level)
    mask = mask.astype("uint32")
    mask = ants.from_numpy(mask)
    mask.set_direction(target_image.direction)
    mask.set_spacing(target_image.spacing)
    mask.set_origin(target_image.origin)

    pbar.update(1)

    pbar.set_postfix({"step": "Aligning annotations to volume"})
    # Align the annotations to the volume
    nodes = load_zarr(zarr_file)
    target_image = load_ants_image_from_node(nodes[0], resolution_level, channel=0)
    template = ants.image_read(template_path)

    if not use_custom_atlas:
        # TODO: Fix the circular import caused by importing download_allen_atlas
        # base_atlas, _ = download_allen_atlas(temp_dir.name, resolution=atlas_resolution, keep_nrrd=False)
        pass
    else:
        base_atlas = ants.image_read(atlas_path)

    atlas = align_annotations_to_volume(target_volume=target_image, mask=mask, template=template, atlas=base_atlas,
                                        resolution=25, keep_intermediary=False, data_dir=temp_dir.name)

    # Reorient the atlas to the same orientation as the target image
    atlas = ants.reorient_image2(atlas, target_image.orientation)

    # Resize the atlas to full size
    atlas_target_shape = nodes[0].data[0].shape
    if len(atlas_target_shape) == 4:
        atlas_target_shape = atlas_target_shape[1:]
    atlas = da.from_array(atlas, chunks=(128, 128, 128))
    atlas_resized = da.transpose(atlas, (2, 1, 0))
    atlas_resized = resize(atlas_resized, atlas_target_shape, order=0)

    save_atlas_to_zarr(zarr_file, atlas_resized, scales=scales, chunks=chunks, resolution_level=resolution_level)
    temp_dir.cleanup()
    pbar.update(1)

    # Creating final mask
    pbar.set_postfix({"step": "Creating final mask"})
    nodes = load_zarr(zarr_file)
    atlas_node = load_node_by_name(nodes, "atlas")
    atlas = load_zarr_image_from_node(atlas_node, 0)

    # Set all non-zero pixels of the atlas to 1
    atlas[atlas > 0] = 1

    # Save to zarr
    atlas = atlas.astype("int8")
    color_dict = generate_label_color_dict_mask()
    save_label_to_zarr(atlas, zarr_file, scales=scales, chunks=chunks, color_dict=color_dict,
                       name="mask", resolution_level=resolution_level)
    pbar.update(1)

    pbar.set_postfix({"step": "Done"})
    pbar.update(1)
    pbar.close()
