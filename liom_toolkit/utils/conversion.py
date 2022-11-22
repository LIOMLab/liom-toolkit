import os

import h5py as h5
import nibabel as nib
import nrrd
import numpy as np
import tqdm as tqdm
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


def get_first_key(file):
    """
    Get the first key in a HDF5 file.
    :param file: The HDF5 file.
    :return: The first key. Useful for getting the shape of the data.
    """
    keys = file.keys()
    for first_key in keys:
        first_key = first_key
        break
    return first_key


def load_hdf5(hdf5_file, map_file="temp.dat"):
    """
    Load a HDF5 into a numpy memmap.
    :param hdf5_file: The HDF5 file to load.
    :param map_file: The memmap file to save to.
    :return: The memmap.
    """
    with h5.File(hdf5_file) as file:
        keys = file.keys()
        n_frames = len(keys)
        key = get_first_key(file)
        frame = file[key][:]
        data = np.memmap(map_file, dtype=np.uint16, mode='w+', shape=(n_frames, frame.shape[0], frame.shape[1]))
        for i, key in enumerate(tqdm.tqdm(keys, desc="Converting HDF5 to zarr file")):
            frame = file[key][:]
            data[i, :, :] = frame
    return data


def convert_hdf5_to_nifti(hdf5_file, nifti_file):
    """
    Convert a HDF5 file from the lightsheet microscope to a NIFTI file.
    :param hdf5_file: Path to the HDF5 file.
    :param nifti_file: Path to the NIFTI file.
    :return:
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, map_file)

    print("Transposing...")
    data = np.transpose(data, (1, 2, 0))
    print("Saving...")
    ni_img = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.uint16)
    nib.save(ni_img, nifti_file)
    print("Done!")
    os.remove(map_file)


def save_zarr(data, zarr_file, axes="zxy", chunks=(1, 2048, 2048)):
    """
    Save a numpy array to a zarr file.
    :param data: The data to save.
    :param zarr_file: The zarr file to save to.
    :param axes: The order of the axes of the data.
    :param chunks: The chunk size to use.
    :return:
    """
    print("Saving...")
    os.mkdir(zarr_file)
    store = parse_url(zarr_file, mode="w").store
    root = zarr.group(store=store)
    write_image(image=data, group=root, axes=axes, storage_options=dict(chunks=chunks))
    print("Done!")


def convert_hdf5_to_zarr(hdf5_file, zarr_file):
    """
    Convert a HDF5 file from the lightsheet microscope to a zarr file.
    :param hdf5_file: Path to the HDF5 file.
    :param zarr_file: Path to the zarr file.
    :return:
    """
    map_file = "temp.dat"
    data = load_hdf5(hdf5_file, map_file)

    save_zarr(data, zarr_file, chunks=(1, 2048, 2048))
    os.remove(map_file)


def convert_nifti_to_zarr(nifti_file, zarr_file, shape=None):
    """
    Convert a NIFTI file to a zarr file.
    :param nifti_file: The NIFTI file to convert.
    :param zarr_file: The zarr file to save to.
    :param shape: The shape of the data. If None, the shape will be inferred from the NIFTI file.
    """
    print("Loading...")
    ni_img = nib.load(nifti_file)
    data = ni_img.get_fdata()
    if shape is None:
        shape = data.shape
    save_zarr(data, zarr_file, axes="xyz", chunks=shape)


def convert_nrrd_to_zarr(nrrd_file, zarr_file, shape=None):
    """
    Convert a NRRD file to a zarr file.
    :param nrrd_file: The NRRD file to convert.
    :param zarr_file: The zarr file to save.
    :param shape: The shape of the data to save. If None, the shape of the data will be used.
    """
    print("Loading...")
    data, header = nrrd.read(nrrd_file)
    if shape is None:
        shape = data.shape
    save_zarr(data, zarr_file, axes="xyz", chunks=shape)
